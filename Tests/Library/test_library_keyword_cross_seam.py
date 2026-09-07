"""The four-seam keyword merge must rank fairly across seams (TASK-16071).

The Library's plain keyword path fans out over four seams (notes, media,
conversations, prompts) and used to merge them by CONCATENATION in that
fixed source order. Nothing carried a score and nothing sorted, so a row's
cross-seam position was decided by its SOURCE TYPE and by how many rows the
earlier seams happened to return -- never by how well it matched. Any pass
that matched `top_k` notes therefore buried every media, conversation and
prompt hit behind them, and every downstream cut (the evidence list the user
reads, RAG Answer's evidence budget, a harness's doc-level k) cut exactly
there.

Measured in TASK-16071's filing: with an oracle-fed widened pass, media
targets landed at merged position 14 behind 13 notes and conversation
targets at 19-21 -- displacement, not non-match (every one of them was
present at a deeper k). Prompts, the fourth and last seam, were the most
buried of all.

The pins below are all on the REAL path with REAL databases (the fixture
idiom of `test_library_local_rag_search_service.py`'s `real_fts_app` /
`real_prompts_app`, extended to all four seams at once), because the defect
lives in the interaction between four seams' independent `limit=top_k` caps
and the merge, and no double reproduces that:

(a) displacement -- a media seam's rank-1 row must precede the notes seam's
    rank-5 row (RED on concatenation);
(b) rank-fairness -- four equally-sized seams alternate by POSITION, and
    seam order breaks ties WITHIN a position (that tie-break is the pinned
    semantics, not an accident: see the test's own docstring);
(c) single-seam byte-identity -- a query only the notes seam matches
    produces exactly the list it always did;
(d) the no-truncation contract -- every row each seam returns is still in
    the merged list; this site cuts nothing, consumers do;
(e) prompts-seam participation -- the fourth, most-buried seam interleaves
    with the rest instead of being appended last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)

#: The term every seeded document in every seam carries, so one query
#: reaches all four seams at once.
TERM = "kestrel"

#: A term only the notes seam's documents carry -- the single-seam pin (c).
NOTES_ONLY_TERM = "peregrine"

#: All four Library source-type identifiers, in the seam order the merge
#: iterates (`_KNOWN_KEYWORD_SOURCE_TYPES`). Spelled literally rather than
#: imported so this file pins the contract instead of restating whatever
#: the implementation happens to define.
ALL_SOURCES = ("notes", "media", "conversations", "prompts")

#: The `provenance.source_type` each seam's row builder stamps -- SINGULAR,
#: unlike the plural scope identifiers above.
NOTE, MEDIA, CONVERSATION, PROMPT = "note", "media", "conversation", "prompt"

_USER_ID = "cross-seam-user"
_CLIENT_ID = "cross-seam-client"


@dataclass
class FourSeamFixture:
    """A live four-seam app double over four real, seeded databases."""

    app: SimpleNamespace
    note_ids: list[str] = field(default_factory=list)
    media_ids: list[str] = field(default_factory=list)
    conversation_ids: list[str] = field(default_factory=list)
    prompt_ids: list[str] = field(default_factory=list)


def _body(marker: str, *, notes_only: bool = False) -> str:
    """A short body carrying `TERM` (and `NOTES_ONLY_TERM` for notes)."""
    extra = f" The {NOTES_ONLY_TERM} log mentions it too." if notes_only else ""
    return (
        f"Maintenance record {marker}: the {TERM} assembly was inspected "
        f"and returned to service.{extra}"
    )


@pytest.fixture
def four_seams(tmp_path) -> Callable[..., FourSeamFixture]:
    """Build a four-seam app over real DBs with per-seam document counts.

    Real databases rather than doubles on purpose: the behaviour under test
    is what happens when four INDEPENDENTLY capped seams are merged, and a
    double that hands back a fixed list cannot show a cap interacting with
    a merge. The four writers are the ones production uses
    (`NotesInteropService.add_note`, `MediaDatabase.add_media_with_keywords`,
    `CharactersRAGDB.add_message`, `PromptsDatabase.add_prompt`), so a row
    that is written is a row that is searchable, by construction.
    """
    closers: list[Callable[[], None]] = []

    def build(
        *, notes: int = 0, media: int = 0, conversations: int = 0, prompts: int = 0
    ) -> FourSeamFixture:
        chachanotes_path = tmp_path / "cross_seam_chachanotes.db"
        media_path = tmp_path / "cross_seam_media.db"
        prompts_path = tmp_path / "cross_seam_prompts.db"
        notes_dir = tmp_path / "cross_seam_notes"
        notes_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        chachanotes_db = CharactersRAGDB(chachanotes_path, client_id=_CLIENT_ID)
        closers.append(chachanotes_db.close_connection)
        media_db = MediaDatabase(media_path, client_id=_CLIENT_ID)
        closers.append(media_db.close_connection)
        prompts_db = PromptsDatabase(prompts_path, client_id=_CLIENT_ID)
        closers.append(prompts_db.close_connection)
        notes_service = NotesInteropService(
            base_db_directory=notes_dir,
            api_client_id=_CLIENT_ID,
            global_db_to_use=chachanotes_db,
        )
        # `NotesInteropService` opens its own per-user handle onto the same
        # file rather than reusing the template object, so those are ours to
        # close as well -- but never the template twice.
        closers.append(
            lambda: [
                db.close_connection()
                for db in notes_service._db_instances.values()
                if db is not chachanotes_db
            ]
        )

        fixture = FourSeamFixture(
            app=SimpleNamespace(
                notes_scope_service=NotesScopeService(
                    local_notes_service=notes_service, server_service=None
                ),
                notes_user_id=_USER_ID,
                media_reading_scope_service=MediaReadingScopeService(
                    LocalMediaReadingService(media_db), None
                ),
                chachanotes_db=chachanotes_db,
                prompt_scope_service=PromptScopeService(
                    local_service=LocalPromptService(prompts_db),
                    server_service=None,
                ),
            )
        )

        for index in range(notes):
            note_id = notes_service.add_note(
                _USER_ID,
                f"Note {index} inspection",
                _body(f"N{index}", notes_only=True),
            )
            assert note_id, f"note {index} write failed"
            fixture.note_ids.append(str(note_id))
        for index in range(media):
            media_id, _uuid, message = media_db.add_media_with_keywords(
                title=f"Media {index} inspection",
                media_type="document",
                content=_body(f"M{index}"),
            )
            assert media_id is not None, f"media {index} write failed: {message}"
            fixture.media_ids.append(str(media_id))
        for index in range(conversations):
            conversation_id = chachanotes_db.add_conversation(
                {"title": f"Conversation {index} inspection"}
            )
            assert conversation_id, f"conversation {index} write failed"
            chachanotes_db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": _body(f"C{index}"),
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
            fixture.conversation_ids.append(str(conversation_id))
        for index in range(prompts):
            prompt_id, _uuid, message = prompts_db.add_prompt(
                name=f"Prompt {index} inspection",
                author="tester",
                details=None,
                system_prompt=_body(f"P{index}"),
            )
            assert prompt_id is not None, f"prompt {index} write failed: {message}"
            fixture.prompt_ids.append(str(prompt_id))

        return fixture

    try:
        yield build
    finally:
        for close in reversed(closers):
            close()


async def _search(app, query: str, sources=ALL_SOURCES, *, top_k: int = 5):
    """Run the production keyword path and return its merged rows."""
    result = await LibraryLocalRagSearchService(app).search(
        query, tuple(sources), "search", top_k=top_k
    )
    assert isinstance(result, dict), f"expected rows, got outcome: {result!r}"
    return list(result["results"])


async def _seam_ranking(app, source: str, query: str, *, top_k: int = 5):
    """One seam's OWN ranking, read from the seam method, not from a search.

    The reference order has to come from upstream of the merge. Reading it
    back with a single-source `search()` call looks equivalent and is not:
    that path runs the merge too, so a merge defect that permutes a seam's
    own rows permutes the reference identically and the comparison passes
    against itself. Caught exactly that way -- a mutation reversing each
    seam's ranking before the interleave went undetected until this helper
    stopped going through `search()`.

    THE TRADE (review, 2026-08-14): this helper hand-duplicates
    `_search_keyword`'s seam-call recipe instead of sharing code with it.
    Today the duplication is exact -- `outcomes[source_type][1]` reaches
    `interleave_rankings` with NO transformation between the seam call and
    the merge -- so the reference is byte-identical to what the merge
    consumes. If a future change inserts ANY post-seam, pre-merge step in
    `_search_keyword` (a filter, a threaded kwarg, a normalization), this
    helper goes silently stale: update it in the same commit, or the pins
    compare against a reference the merge no longer sees.
    """
    service = LibraryLocalRagSearchService(app)
    if source == "notes":
        available, rows = await service._search_notes(
            query, top_k, getattr(app, "notes_user_id", "default_user")
        )
    elif source == "media":
        available, rows = await service._search_media(query, top_k)
    elif source == "conversations":
        available, rows = await service._search_conversations(query, top_k)
    elif source == "prompts":
        available, rows = await service._search_prompts(query, top_k)
    else:  # pragma: no cover - the four seams are the whole vocabulary
        raise AssertionError(f"unknown seam {source!r}")
    assert available, f"the {source} seam reported itself unavailable"
    return rows


def _keys(rows) -> list[tuple[str, str]]:
    """The `(source_type, source_id)` identity of each row, in list order."""
    return [(row["provenance"]["source_type"], row["source_id"]) for row in rows]


def _types(rows) -> list[str]:
    return [row["provenance"]["source_type"] for row in rows]


@pytest.mark.asyncio
async def test_a_media_rank_one_row_precedes_the_notes_seams_fifth_row(four_seams):
    """(a) THE DISPLACEMENT PIN -- the defect, stated as an ordering.

    Five notes, one media document, all matching the same query at
    `top_k=5`. Under the fixed-order concatenation the media seam's BEST row
    lands at merged position 6, behind all five notes, purely because
    "notes" is iterated first -- so a consumer taking the top five never
    sees it. Rank-fair merging puts each seam's rank-1 row in the first
    round, ahead of every seam's rank-2-and-deeper rows.
    """
    fixture = four_seams(notes=5, media=1)

    rows = await _search(fixture.app, TERM)
    keys = _keys(rows)
    notes_rows = _keys(await _seam_ranking(fixture.app, "notes", TERM))
    media_rows = _keys(await _seam_ranking(fixture.app, "media", TERM))
    assert len(notes_rows) == 5, notes_rows
    assert len(media_rows) == 1, media_rows

    media_rank_one = keys.index(media_rows[0])
    notes_rank_five = keys.index(notes_rows[4])
    assert media_rank_one < notes_rank_five, (
        "the media seam's rank-1 row is merged BEHIND the notes seam's "
        f"rank-5 row (media at {media_rank_one}, notes#5 at "
        f"{notes_rank_five}); cross-seam position is being decided by seam "
        f"order, not by rank. Merged order: {_types(rows)}"
    )


@pytest.mark.asyncio
async def test_b_equal_seams_alternate_by_position_in_fixed_seam_order(four_seams):
    """(b) RANK-FAIRNESS, and the order-WITHIN-a-position semantics.

    Four seams with two matching documents each: the merged list must be
    one row from every seam, then the second row from every seam. The pin
    also fixes what happens INSIDE a position, where there is genuinely no
    signal to choose on -- raw FTS5 scores from four different tables are
    not comparable (the engine rejected a cross-seam score merge for this
    reason), so ties are broken by the seam order
    `_KNOWN_KEYWORD_SOURCE_TYPES` declares. That is a documented, pinned
    convention rather than a claim about relevance: within one position,
    notes still precede media, which precede conversations, which precede
    prompts.

    Two properties, because the seam-type cycle alone does not pin the
    merge: a round-robin that shuffled each seam's own rows would still
    produce the right sequence of TYPES. The second assertion is the one
    that fails then -- each seam's rows must appear in the merged list in
    exactly the relative order that seam returned them.
    """
    fixture = four_seams(notes=2, media=2, conversations=2, prompts=2)

    rows = await _search(fixture.app, TERM)

    assert _types(rows) == [
        NOTE,
        MEDIA,
        CONVERSATION,
        PROMPT,
        NOTE,
        MEDIA,
        CONVERSATION,
        PROMPT,
    ], f"seams did not alternate by position: {_types(rows)}"

    merged = _keys(rows)
    for source, source_type in zip(ALL_SOURCES, (NOTE, MEDIA, CONVERSATION, PROMPT)):
        own = _keys(await _seam_ranking(fixture.app, source, TERM))
        assert len(own) == 2, (source, own)
        in_merged = [key for key in merged if key[0] == source_type]
        assert in_merged == own, (
            f"the {source} seam's rows were reordered by the merge: seam order "
            f"{own}, merged order {in_merged}"
        )


@pytest.mark.asyncio
async def test_c_a_single_seam_query_returns_a_byte_identical_list(four_seams):
    """(c) SINGLE-SEAM BYTE-IDENTITY -- nothing changes with one seam.

    `NOTES_ONLY_TERM` appears in the notes bodies alone, so three of the
    four seams contribute nothing. A round-robin over one non-empty ranking
    is that ranking, and this pins it: the rows, their order and their
    contents are exactly what a notes-only search produces.
    """
    fixture = four_seams(notes=4, media=2, conversations=2, prompts=2)

    merged = await _search(fixture.app, NOTES_ONLY_TERM)
    notes_only = await _search(fixture.app, NOTES_ONLY_TERM, ("notes",))

    assert merged == notes_only, "a single-seam query's list changed"
    assert len(merged) == 4, merged
    assert set(_types(merged)) == {NOTE}, _types(merged)


@pytest.mark.asyncio
async def test_d_the_merge_truncates_nothing(four_seams):
    """(d) THE NO-TRUNCATION CONTRACT -- this site cuts nothing.

    Each seam is capped at `top_k` INDEPENDENTLY, so a four-seam query can
    legitimately return up to `4 * top_k` rows and always has; the cut
    belongs to the consumers (the evidence list, RAG Answer's budget). A
    rank-fair merge changes ORDER only, and this pin is what stops a future
    edit from quietly adding a cross-seam cap here -- which would look like
    an improvement and would silently drop the very rows the reordering
    exists to surface.
    """
    top_k = 5
    fixture = four_seams(notes=6, media=6, conversations=6, prompts=6)

    rows = await _search(fixture.app, TERM, top_k=top_k)

    per_seam = {}
    for source in ALL_SOURCES:
        per_seam[source] = _keys(
            await _seam_ranking(fixture.app, source, TERM, top_k=top_k)
        )
        assert len(per_seam[source]) == top_k, (source, per_seam[source])

    assert len(rows) == 4 * top_k, (
        f"merged {len(rows)} rows from four seams of {top_k}; the merge must "
        "not truncate"
    )
    expected = {key for keys in per_seam.values() for key in keys}
    assert set(_keys(rows)) == expected
    assert len(set(_keys(rows))) == len(rows), "the merge emitted a duplicate row"


@pytest.mark.asyncio
async def test_d2_rows_with_an_empty_source_id_are_not_collapsed(four_seams):
    """(d2) THE DEGENERATE-KEY ARM of the no-truncation contract.

    `interleave_rankings` dedups on ONE `seen` set spanning the whole merge.
    The site argues cross-seam collisions are structurally impossible because
    the seams are disjoint by source type -- true for DISTINCT ids, and
    silent about MISSING ones. Every row builder falls back to `""` when its
    id is absent (`str(item.get("id", ""))` and siblings), and the prompts
    normalizer yields `local_id=None` for any non-local backend, so a future
    change threading a non-local prompt mode through this path would make
    every such row after the first collide on `("prompt", "")` and vanish --
    a silent truncation at the one site whose comment promises it truncates
    nothing. Pinned at the merge helper directly: the fixtures upstream are
    all well-formed by construction, which is exactly why (d) cannot see it.
    """
    from tldw_chatbook.RAG_Search.fusion import interleave_rankings
    from tldw_chatbook.Library.library_local_rag_search_service import (
        _keyword_row_identity,
    )

    def _row(source_type: str, source_id: str, title: str) -> dict:
        return {
            "source_id": source_id,
            "chunk_id": "",
            "title": title,
            "snippet": "",
            "score": None,
            "provenance": {"source_type": source_type},
        }

    degenerate = [
        [_row("prompt", "", "first"), _row("prompt", "", "second")],
        [_row("note", "7", "note seven")],
    ]

    # The PRODUCTION key, not a lambda restating it -- the hole lives in
    # what the merge site actually passes.
    merged = interleave_rankings(degenerate, key=_keyword_row_identity)

    titles = [row["title"] for row in merged]
    assert titles.count("second") == 1, (
        "a row whose source_id is empty was collapsed into its sibling: "
        f"{titles}. The merge site promises no truncation; an id-less row "
        "must keep its slot (give the key a positional tiebreak, or make the "
        "builders refuse an empty source_id)."
    )
    assert len(merged) == 3, titles


@pytest.mark.asyncio
async def test_e_the_prompts_seam_participates_instead_of_being_appended_last(
    four_seams,
):
    """(e) PROMPTS-SEAM PARTICIPATION -- the fourth and most buried seam.

    Prompts are iterated last, so under concatenation a prompt row sat
    behind every note, media and conversation row in the pass -- the reason
    TASK-16071's filing calls it "the buried fourth seam". With five notes
    ahead of it, its single hit landed at merged position 8. Rank-fair
    merging puts it in the first round, ahead of the notes seam's SECOND
    row.
    """
    fixture = four_seams(notes=5, media=1, conversations=1, prompts=1)

    rows = await _search(fixture.app, TERM)
    keys = _keys(rows)
    notes_rows = _keys(await _seam_ranking(fixture.app, "notes", TERM))
    prompt_rows = _keys(await _seam_ranking(fixture.app, "prompts", TERM))
    assert len(prompt_rows) == 1, prompt_rows

    prompt_position = keys.index(prompt_rows[0])
    notes_rank_two = keys.index(notes_rows[1])
    assert prompt_position < notes_rank_two, (
        f"the prompts seam's only hit is merged at {prompt_position}, behind "
        f"the notes seam's rank-2 row at {notes_rank_two}: the fourth seam is "
        f"still being appended rather than interleaved. Merged order: "
        f"{_types(rows)}"
    )
    assert _types(rows)[:4] == [NOTE, MEDIA, CONVERSATION, PROMPT], _types(rows)
