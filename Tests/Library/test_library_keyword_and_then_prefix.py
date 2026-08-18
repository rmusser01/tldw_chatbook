"""The four-seam keyword path runs ``and_then_prefix`` (TASK-17755).

TASK-3997 measured what the Library's plain Search path's AND-strictness
costs on the gated corpus (172 docs, 53 ground-truthed golden queries) and
the owner ruled on 2026-08-18:

| construction | MRR | zero-row queries |
|---|---|---|
| AND-strict (shipped until now) | 0.396 | 32 of 53 |
| pure prefix OR | 0.261 | 0 |
| **and_then_prefix** | **0.423** | 25 |

The naive fix -- replace AND with OR/prefix outright -- was measured *worse*
than the status quo, because 20-30 loosely-matching rows per query on a
172-document corpus bury the answers AND was already getting right.
``and_then_prefix`` beats both, and the reason is structural rather than
statistical: **a sub-leg whose AND primary returned rows never runs the
widening form at all**, so the only results it can change are the ones that
were empty. That property is what makes this a low-risk change, and it is
the first thing pinned below -- pinned by asserting the prefix form is never
*built*, not merely that the rows happen to come out the same. A test that
only compares output cannot tell "the fallback correctly did not fire" from
"the fallback fired and happened to return the same rows".

The five pins:

(a) **sub-leg byte-identity** -- a seam whose primary returns rows returns
    exactly the rows it returned before, and the prefix form is never built;
(b) **query-level byte-identity** -- a four-seam query where every seam's
    primary returns rows is untouched, prefix form never built, no extra
    query issued to any seam;
(c) **the zero-row rescue** -- a seam whose primary returns nothing runs the
    prefix form and comes back with rows;
(d) **per-sub-leg independence** -- one query, one seam whose primary hits
    and three whose primaries miss: the hitter keeps its AND rows untouched
    while the other three are rescued. The decision is per sub-leg, exactly
    as the engine makes it (`RAGService._fts_rows_with_fallback`), never
    per query;
(e) **the merge contract survives** -- TASK-16071's rank-fair round-robin
    (`interleave_rankings` keyed on `_keyword_row_identity`) still holds
    with fallback rows in the mix: every seam's rows are present, nothing is
    truncated, and position-then-seam-order still decides.

Real databases throughout, for the reason TASK-16071's suite gives: the
behaviour under test is what four independently-capped seams do when merged,
and no double reproduces that. The vocabulary is chosen so the AND primary's
own plural/singular widening cannot reach the rescue documents -- "tension"
does not widen to "tensioner" -- which is what makes a prefix rescue
distinguishable from the widening the path already had.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Library import library_local_rag_search_service as seam_module
from tldw_chatbook.Library.library_fts_query import (
    build_fts_match_query,
    build_prefix_match_query,
)
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

#: A term every seeded document carries, so a one-term query reaches all four
#: seams' primaries at once (pins a and b).
SHARED = "kestrel"

#: Present verbatim in the NOTES document only.
EXACT = "tension"

#: Present in the media/conversations/prompts documents only. It begins with
#: `EXACT`, so a PREFIX form reaches it -- while `build_fts_match_query`'s
#: plural/singular widening ("tension" -> "tensions") provably cannot. That
#: gap is the whole experiment: a rescue here is a rescue by the new form,
#: not by widening the path already shipped.
PREFIXED = "tensioner"

#: Notes' primary hits, the other three seams' primaries return zero.
SPLIT_QUERY = f"{SHARED} {EXACT}"

#: Every seam's primary hits: the untouched-by-construction case.
ALL_PRIMARY_QUERY = SHARED

ALL_SOURCES = ("notes", "media", "conversations", "prompts")
NOTE, MEDIA, CONVERSATION, PROMPT = "note", "media", "conversation", "prompt"

_USER_ID = "prefix-fallback-user"
_CLIENT_ID = "prefix-fallback-client"

def _notes_body(marker: str) -> str:
    """A notes body carrying `EXACT` verbatim -- reachable by the AND primary."""
    return (
        f"Inspection record {marker}: the {SHARED} gauge logs {EXACT} "
        f"across the mast."
    )


def _prefixed_body(marker: str) -> str:
    """A body carrying only `PREFIXED` -- reachable ONLY by the prefix form.

    The `marker` is not decoration: the media writer rejects a second
    document with identical content as a duplicate ("Overwrite not
    enabled"), so seeding several rows per seam requires distinct bodies.
    """
    return (
        f"Inspection record {marker}: the {SHARED} bracket carries a "
        f"{PREFIXED} and a shim."
    )


@dataclass
class Seams:
    """A live four-seam app double over four real, seeded databases."""

    app: SimpleNamespace


@pytest.fixture
def seams(tmp_path) -> Callable[..., Seams]:
    """Seed the four seams with per-seam bodies and return the app double.

    Distinct from `test_library_keyword_cross_seam.py`'s `four_seams`, which
    gives every seam the same body on purpose (its subject is the merge, so
    every seam must match every query). This one's subject is *divergence*
    between seams under one query, so the bodies have to differ.
    """
    closers: list[Callable[[], None]] = []

    def build(*, notes: int = 1, media: int = 1, conversations: int = 1,
              prompts: int = 1) -> Seams:
        notes_dir = tmp_path / "prefix_notes"
        notes_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        chachanotes_db = CharactersRAGDB(
            tmp_path / "prefix_chachanotes.db", client_id=_CLIENT_ID
        )
        closers.append(chachanotes_db.close_connection)
        media_db = MediaDatabase(tmp_path / "prefix_media.db", client_id=_CLIENT_ID)
        closers.append(media_db.close_connection)
        prompts_db = PromptsDatabase(
            tmp_path / "prefix_prompts.db", client_id=_CLIENT_ID
        )
        closers.append(prompts_db.close_connection)
        notes_service = NotesInteropService(
            base_db_directory=notes_dir,
            api_client_id=_CLIENT_ID,
            global_db_to_use=chachanotes_db,
        )
        closers.append(
            lambda: [
                db.close_connection()
                for db in notes_service._db_instances.values()
                if db is not chachanotes_db
            ]
        )

        for index in range(notes):
            assert notes_service.add_note(
                _USER_ID, f"Record N{index}", _notes_body(f"N{index}")
            ), f"note {index} write failed"
        for index in range(media):
            media_id, _uuid, message = media_db.add_media_with_keywords(
                title=f"Record M{index}",
                media_type="document",
                content=_prefixed_body(f"M{index}"),
            )
            assert media_id is not None, f"media {index} write failed: {message}"
        for index in range(conversations):
            conversation_id = chachanotes_db.add_conversation(
                {"title": f"Record C{index}"}
            )
            assert conversation_id, f"conversation {index} write failed"
            chachanotes_db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": _prefixed_body(f"C{index}"),
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
        for index in range(prompts):
            prompt_id, _uuid, message = prompts_db.add_prompt(
                name=f"Record P{index}",
                author="tester",
                details=None,
                system_prompt=_prefixed_body(f"P{index}"),
            )
            assert prompt_id is not None, f"prompt {index} write failed: {message}"

        return Seams(
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

    try:
        yield build
    finally:
        for close in reversed(closers):
            close()


@pytest.fixture
def prefix_spy(monkeypatch):
    """Count every construction of the PREFIX form, at the CONSUMER namespace.

    Patched on `library_local_rag_search_service`, never on the defining
    module. TASK-3997's own A/B run was invalidated exactly once by getting
    this wrong: the seam does `from ... import build_prefix_match_query`,
    which binds the name at import time, so patching the definition site
    changes nothing and the "intervention" silently does nothing -- which
    produces a perfect-looking null rather than an error.

    Guarded against that here: the fixture asserts the name it is replacing
    is actually the one the seam module holds, so a future refactor that
    stops importing it this way fails loudly instead of neutering the spy.
    """
    assert seam_module.build_prefix_match_query is build_prefix_match_query, (
        "the seam module no longer holds the imported prefix builder; this "
        "spy would patch a name nothing calls"
    )

    calls: list[str] = []

    def spy(query: str) -> str:
        calls.append(query)
        return build_prefix_match_query(query)

    monkeypatch.setattr(seam_module, "build_prefix_match_query", spy)
    return calls


async def _search(app, query: str, sources=ALL_SOURCES, *, top_k: int = 5):
    """Run the production keyword path and return its merged rows."""
    result = await LibraryLocalRagSearchService(app).search(
        query, tuple(sources), "search", top_k=top_k
    )
    assert isinstance(result, dict), f"expected rows, got outcome: {result!r}"
    return list(result["results"])


def _keys(rows) -> list:
    """Each row's identity, read through the merge's OWN key function.

    `_keyword_row_identity`, not a hand-rolled `(type, id)` tuple: pin (e)
    is a claim about what `interleave_rankings` dedups on, and restating the
    key here would let the two drift while the pin kept passing.
    """
    return [seam_module._keyword_row_identity(row) for row in rows]


def _types(rows) -> list[str]:
    return [row["provenance"]["source_type"] for row in rows]


# --- (a) sub-leg byte-identity: the property that makes this low-risk -------


@pytest.mark.asyncio
async def test_a_sub_leg_whose_primary_returns_rows_never_builds_the_prefix_form(
    seams, prefix_spy
):
    """AC#2 at the sub-leg. The rescue path is not merely unused -- unbuilt.

    Asserting on the ROWS alone would pass just as happily if the fallback
    had run and returned the same document, which is the failure mode that
    would make this change risky without anyone noticing. So the assertion
    is on the construction counter.
    """
    app = seams().app
    service = LibraryLocalRagSearchService(app)

    available, rows = await service._search_notes(SPLIT_QUERY, 5, _USER_ID)

    assert available is True
    assert len(rows) == 1, f"expected the notes primary to hit: {rows!r}"
    assert prefix_spy == [], (
        "the prefix form was built for a sub-leg whose primary returned "
        f"rows: {prefix_spy!r}"
    )


@pytest.mark.asyncio
async def test_a_hitting_sub_legs_rows_are_byte_identical_to_the_and_only_answer(
    seams,
):
    """The rows themselves, compared against the AND expression run directly.

    The reference is read UPSTREAM of the seam method -- straight from the
    notes service with `build_fts_match_query`'s expression -- rather than
    from a second `_search_notes` call, so the comparison cannot pass
    against itself the way a same-path reference would.
    """
    app = seams().app
    service = LibraryLocalRagSearchService(app)

    _available, rows = await service._search_notes(SPLIT_QUERY, 5, _USER_ID)
    reference = await app.notes_scope_service.search_notes(
        scope="local_note",
        query=SPLIT_QUERY,
        limit=5,
        user_id=_USER_ID,
        fts_match_query=build_fts_match_query(SPLIT_QUERY),
    )

    assert _keys(rows) == [
        seam_module._keyword_row_identity(seam_module._note_row(item))
        for item in reference
    ]


# --- (b) query-level byte-identity: no fallback, no extra query -------------


@pytest.mark.asyncio
async def test_a_query_every_seam_matches_is_untouched_by_the_construction(
    seams, prefix_spy
):
    """AC#2 at the query. All four primaries hit; nothing widens."""
    app = seams().app

    rows = await _search(app, ALL_PRIMARY_QUERY)

    assert sorted(_types(rows)) == sorted(
        [NOTE, MEDIA, CONVERSATION, PROMPT]
    ), f"every seam should have contributed its primary row: {_types(rows)!r}"
    assert prefix_spy == [], (
        f"a fallback was built for an all-primary query: {prefix_spy!r}"
    )


@pytest.mark.asyncio
async def test_an_all_primary_query_issues_exactly_one_match_per_seam(seams):
    """"No extra query" is a claim about the DATABASE, so count DB calls.

    Wraps the notes seam's own service call. A fallback that never fires
    still costs nothing only if the second MATCH is never issued.
    """
    app = seams().app
    real_search = app.notes_scope_service.search_notes
    issued: list[str] = []

    async def counting_search(**kwargs):
        issued.append(kwargs.get("fts_match_query"))
        return await real_search(**kwargs)

    app.notes_scope_service.search_notes = counting_search

    _available, rows = await LibraryLocalRagSearchService(app)._search_notes(
        ALL_PRIMARY_QUERY, 5, _USER_ID
    )

    assert rows, "the primary was supposed to hit"
    assert issued == [build_fts_match_query(ALL_PRIMARY_QUERY)], (
        f"expected exactly the AND primary and nothing else: {issued!r}"
    )


# --- (c) the zero-row rescue ------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seam, source_type",
    [("media", MEDIA), ("conversations", CONVERSATION), ("prompts", PROMPT)],
)
async def test_a_zero_row_sub_leg_is_rescued_by_the_prefix_form(
    seams, prefix_spy, seam, source_type
):
    """AC#1/AC#3. The document is unreachable under AND and reachable now."""
    app = seams().app
    service = LibraryLocalRagSearchService(app)
    call = {
        "media": lambda: service._search_media(SPLIT_QUERY, 5),
        "conversations": lambda: service._search_conversations(SPLIT_QUERY, 5),
        "prompts": lambda: service._search_prompts(SPLIT_QUERY, 5),
    }[seam]

    available, rows = await call()

    assert available is True
    assert _types(rows) == [source_type], (
        f"the {seam} seam was not rescued: {rows!r}"
    )
    assert prefix_spy == [SPLIT_QUERY], (
        f"expected exactly one prefix build for the empty sub-leg: {prefix_spy!r}"
    )


@pytest.mark.asyncio
async def test_a_query_no_form_can_match_still_returns_nothing(seams, prefix_spy):
    """The rescue is a widening, not a guarantee of rows.

    The fallback runs (the primary was empty) and honestly finds nothing --
    the arm that proves the pin above is measuring a real match rather than
    an unconditional "return something".
    """
    app = seams().app

    rows = await _search(app, "wombat")

    assert rows == []
    assert prefix_spy, "the fallback should have been attempted"


@pytest.mark.asyncio
async def test_an_all_function_word_query_never_runs_an_unbounded_prefix(seams):
    """Trimming to nothing means no rows, never `"the"*` over the corpus.

    A stopword prefix matches "the", "then", "there", "their", "these" --
    nearly a whole corpus -- and the prefix form ANDs its terms, so one such
    term drags the whole expression toward matching everything. The shared
    builder answers "" for that case and the seam must skip the query
    entirely rather than widen to it.
    """
    app = seams().app
    real_search = app.notes_scope_service.search_notes
    issued: list[str] = []

    async def counting_search(**kwargs):
        issued.append(kwargs.get("fts_match_query"))
        return await real_search(**kwargs)

    app.notes_scope_service.search_notes = counting_search

    _available, rows = await LibraryLocalRagSearchService(app)._search_notes(
        "of the", 5, _USER_ID
    )

    assert rows == []
    assert issued == [build_fts_match_query("of the")], (
        f"a second, unbounded MATCH was issued: {issued!r}"
    )


# --- (d) the decision is per sub-leg, not per query -------------------------


@pytest.mark.asyncio
async def test_one_query_mixes_a_primary_sub_leg_with_three_rescued_ones(
    seams, prefix_spy
):
    """AC#1's "per sub-leg" clause, and AC#4's divergence, in one query.

    `SPLIT_QUERY` hits the notes seam's AND primary and misses the other
    three. Per-QUERY fallback would either leave all four alone (no rescue)
    or widen all four (the notes row displaced by loose matches). Per-SUB-
    LEG gives the only correct answer: notes untouched, three rescued.
    """
    app = seams().app

    rows = await _search(app, SPLIT_QUERY)

    assert sorted(_types(rows)) == sorted(
        [NOTE, MEDIA, CONVERSATION, PROMPT]
    ), f"expected one row per seam: {_types(rows)!r}"
    assert len(prefix_spy) == 3, (
        "expected exactly three prefix builds -- one per zero-row sub-leg, "
        f"none for the hitter: {prefix_spy!r}"
    )


@pytest.mark.asyncio
async def test_a_rescued_seam_does_not_change_another_seams_rows(seams):
    """Independence, checked against the seam's own solo answer.

    The notes rows from the four-seam call must equal the notes rows from a
    notes-only call: three other seams falling back around it changes
    nothing about what notes returns.
    """
    app = seams().app

    four_seam_rows = await _search(app, SPLIT_QUERY)
    notes_only_rows = await _search(app, SPLIT_QUERY, sources=("notes",))

    notes_from_four = [
        key for row, key in zip(four_seam_rows, _keys(four_seam_rows))
        if row["provenance"]["source_type"] == NOTE
    ]
    assert notes_from_four == _keys(notes_only_rows)


# --- (e) the TASK-16071 merge contract, with fallback rows present ----------


@pytest.mark.asyncio
async def test_the_rank_fair_merge_still_holds_when_seams_mix_forms(seams):
    """TASK-16071's contract survives the new construction.

    Three seams' rows now arrive from a *fallback* form while one arrives
    from the primary. The merge is `interleave_rankings` keyed on
    `_keyword_row_identity` and knows nothing about forms, so what must
    still hold is exactly what held before: every seam represented, nothing
    truncated, and `_KNOWN_KEYWORD_SOURCE_TYPES` order breaking the tie at
    equal rank.

    NOTE for whoever changes this next: the Library merge is deliberately
    UNTIERED, unlike the engine's (TASK-15700). This test pins the untiered
    order that TASK-3997 measured `and_then_prefix` at. If tiering is ever
    adopted here -- putting whole primary-form sub-legs ahead of fallback
    ones -- this expectation changes and the change must be re-measured, not
    just re-stamped.
    """
    app = seams(notes=3, media=3, conversations=3, prompts=3).app

    rows = await _search(app, SPLIT_QUERY, top_k=3)

    assert len(rows) == 12, f"no seam's rows may be dropped: {_types(rows)!r}"
    assert _types(rows)[:4] == [NOTE, MEDIA, CONVERSATION, PROMPT], (
        f"position 0 of each seam must come first, in seam order: {_types(rows)!r}"
    )
    assert len(set(_keys(rows))) == 12, f"the dedup key collided: {_keys(rows)!r}"


@pytest.mark.asyncio
async def test_rescued_rows_carry_the_same_row_shape_as_primary_rows(seams):
    """A fallback row is a row: same builder, same provenance, same identity.

    Cheap to get wrong (a second code path that hand-builds rows), and it
    would break `_keyword_row_identity` and every downstream consumer
    silently -- the merge would still "work", on garbage keys.
    """
    app = seams().app
    service = LibraryLocalRagSearchService(app)

    _a, primary_rows = await service._search_notes(SPLIT_QUERY, 5, _USER_ID)
    _b, rescued_rows = await service._search_media(SPLIT_QUERY, 5)

    assert primary_rows and rescued_rows
    assert set(primary_rows[0]) == set(rescued_rows[0]), (
        "the rescued row has a different key set than a primary row: "
        f"{sorted(set(primary_rows[0]) ^ set(rescued_rows[0]))!r}"
    )
    assert rescued_rows[0]["provenance"]["source_type"] == MEDIA
    assert rescued_rows[0]["score"] is None, (
        "rescued rows must keep the path's no-score contract, which the "
        "rank-fair merge depends on"
    )
