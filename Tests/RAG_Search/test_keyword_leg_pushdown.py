"""The keyword leg must budget only for the source types the caller asked for.

TASK-14751. TASK-3996 fixed the leg's media-only blindness by round-robining
three sub-legs (media, notes, conversations) into ONE fixed ``top_k`` FTS
budget, while the Library's hybrid arm post-filters the fused rows by the
user's selected source types AFTER fusion. Nothing told the leg which types
the user asked for, so it spent up to two thirds of its budget retrieving
rows that were then discarded downstream.

The reviewer's probe, reproduced verbatim by
``test_none_means_all_three_sub_legs_unchanged`` below: 12 matching
documents of each type seeded, leg asked for 20 -> ``{media 7, note 7,
conversation 6}``. With Media-only selected, 13 of those 20 slots were
thrown away rather than backfilled with media, and the worst case (hybrid +
Media-only + an empty vector index -- the "keyword-only results" route)
showed the user roughly one third of the media rows dev returns.

Two things these tests exist to hold down, and one they deliberately do not:

* **The budget.** A single-type selection gives that sub-leg the FULL
  ``top_k`` (the pre-TASK-3996 behavior for media), through the real hybrid
  path over an empty vector index -- not just through ``_keyword_search``
  directly.
* **Nothing unselected is queried.** Absent rows are not evidence: a
  post-filter produces exactly the same row set while still paying for the
  query. The spies assert the unselected sub-legs never RUN.
* **Not a reversion of interleaving.** FTS5 scores from different tables are
  not comparable, so a multi-type selection must stay rank-fair (round
  robin) rather than concatenating and letting one well-stocked source
  consume every slot.

Real databases throughout (media + ChaChaNotes writer APIs, the
``test_keyword_leg_chacha.py`` pattern). Canned fakes are precisely the
blindness that let this defect live: the Library's unit tests drive doubles,
so no test ever observed the composition of a real leg's output.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

# The query token every seeded document carries.
QUERY = "quokka"


def _make_service(media_db_path=None, chachanotes_db_path=None, enable_cache=False):
    """A RAGService with the in-memory vector store and mock embeddings."""
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = enable_cache
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    if chachanotes_db_path is not None:
        cfg.search.chachanotes_db_path = Path(chachanotes_db_path)
    return RAGService(cfg)


def _seed_media(tmp_path, count, name="tldw_cli_media_v2.db"):
    """Create a real MediaDatabase with `count` matching articles."""
    db_path = tmp_path / name
    db = MediaDatabase(db_path=str(db_path), client_id="test_pushdown")
    try:
        for i in range(count):
            media_id, _, message = db.add_media_with_keywords(
                title=f"Quokka Media {i:02d}",
                content=f"Media article {i} about quokka island populations.",
                media_type="article",
                author="Tester",
                url=f"https://example.com/quokka-media-{i:02d}",
            )
            assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _seed_chacha(tmp_path, notes, conversations, name="chacha.db"):
    """Create a real ChaChaNotes DB with `notes` notes and `conversations` chats."""
    db = CharactersRAGDB(tmp_path / name, "test_pushdown")
    try:
        for i in range(notes):
            assert db.add_note(
                f"Quokka Note {i:02d}",
                f"Field note {i}: quokka smiles are an artifact of jaw shape.",
            )
        for i in range(conversations):
            conv_id = db.add_conversation({"title": f"Quokka Chat {i:02d}"})
            assert conv_id, "conversation seed failed"
            assert db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "user",
                    "content": f"Question {i}: how do quokka joeys forage?",
                }
            ), "message seed failed"
    finally:
        db.close_connection()
    return tmp_path / name


@pytest.fixture
def debug_logs():
    """Collect loguru DEBUG+ records.

    Same idiom as `test_keyword_leg_chacha.warnings_captured`, one level
    lower. `capsys` never sees loguru output, so a "we logged it" claim is
    unverifiable without a sink.
    """
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture
def mixed_corpus(tmp_path):
    """The reviewer's probe corpus: 12 matching documents of each type."""
    media_path = _seed_media(tmp_path, 12)
    chacha_path = _seed_chacha(tmp_path, notes=12, conversations=12)
    return media_path, chacha_path


def _types(results):
    return [r.metadata.get("source_type") for r in results]


def _counts(results):
    out = {}
    for source_type in _types(results):
        out[source_type] = out.get(source_type, 0) + 1
    return out


# --- The budget -------------------------------------------------------------


def test_media_only_selection_gets_the_full_budget(mixed_corpus):
    """AC#2/AC#3 pin: media-only must not be rationed to a third of top_k.

    Reds if the budget silently reverts to a fixed three-way split under a
    single-type selection (7 media instead of all 12).
    """
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    results = asyncio.run(
        service._keyword_search(QUERY, top_k=20, keyword_source_types={"media"})
    )

    assert _counts(results) == {"media": 12}, (
        "media-only selection did not get the full budget: "
        f"{_counts(results)} (a three-way split gives media 7 of 20)"
    )


def test_media_only_hybrid_over_an_empty_vector_index_keeps_the_full_budget(tmp_path):
    """AC#2 literally: the HYBRID path, an empty vector index, media-only.

    This is the route the defect was worst on -- hybrid whose semantic leg
    is empty, so what the user sees IS the keyword leg. Both halves of the
    criterion are exercised here: the scenario runs through
    ``search(search_type="hybrid")`` against an unindexed vector store, and
    the count is compared against the pre-TASK-3996 leg's behavior, which
    gave media the whole budget: N=25 matching media documents, top_k=20 ->
    min(N, top_k) = 20 media rows.

    The `None` control in the same test is the regression itself: with no
    pushdown the same query over the same corpus spends 13 of those 20 slots
    on notes and conversations the caller discards.
    """
    media_path = _seed_media(tmp_path, 25)
    chacha_path = _seed_chacha(tmp_path, notes=12, conversations=12)
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    # The vector store is deliberately never indexed: stats must show it empty
    # so this is genuinely the "semantic leg empty -- keyword-only results"
    # case and not a hybrid whose vector leg quietly carried the ranking.
    assert service.vector_store.get_collection_stats().get("count", 0) == 0

    pushed_down = asyncio.run(
        service.search(
            QUERY, top_k=20, search_type="hybrid", keyword_source_types={"media"}
        )
    )
    assert _counts(pushed_down) == {"media": 20}, (
        "media-only hybrid over an empty vector index returned fewer media "
        f"rows than the pre-TASK-3996 leg did: {_counts(pushed_down)}"
    )

    # Control: the un-pushed leg on the identical corpus/top_k. This is the
    # measured defect, kept in the test so the pin above cannot be satisfied
    # by a corpus that was never big enough to show it.
    unfiltered = asyncio.run(service.search(QUERY, top_k=20, search_type="hybrid"))
    assert _counts(unfiltered).get("media", 0) < 20, (
        "the control lost its meaning: an unselected hybrid search already "
        f"returns a full media budget ({_counts(unfiltered)})"
    )


def test_the_runtime_service_class_forwards_the_selection(tmp_path):
    """`EnhancedRAGServiceV2` -- not `RAGService` -- is what actually runs.

    It overrides `search()` with an explicit signature, so a new base-class
    keyword-only kwarg is not inherited: the Library's pushdown crashed the
    P1 eval harness with `TypeError: EnhancedRAGServiceV2.search() got an
    unexpected keyword argument 'keyword_source_types'` while every unit
    test stayed green, because the doubles mirror `RAGService.search` and
    nothing exercised the class the Library resolves at runtime. This test
    is that missing coverage: the real subclass, the real corpus, the same
    budget assertion.
    """
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
        EnhancedRAGServiceV2,
    )

    media_path = _seed_media(tmp_path, 25)
    chacha_path = _seed_chacha(tmp_path, notes=12, conversations=12)
    cfg = _make_service(
        media_db_path=media_path, chachanotes_db_path=chacha_path
    ).config
    service = EnhancedRAGServiceV2(config=cfg)

    results = asyncio.run(
        service.search(
            QUERY, top_k=20, search_type="hybrid", keyword_source_types={"media"}
        )
    )

    assert _counts(results) == {"media": 20}


# --- Nothing unselected is queried -----------------------------------------


def test_unselected_sub_legs_are_never_queried(mixed_corpus, monkeypatch):
    """Absent rows are not evidence -- the skipped sub-legs must not RUN.

    A post-filter over a full three-way leg produces exactly the same row
    set as a pushdown while still paying for every query it discards, so
    only a spy can tell the two apart.
    """
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    calls: list[str] = []

    original_media = service._media_keyword_subleg

    async def spy_media(*args, **kwargs):
        calls.append("media")
        return await original_media(*args, **kwargs)

    original_notes = RAGService._chacha_notes_fts
    original_conversations = RAGService._chacha_conversations_fts

    def spy_notes(*args, **kwargs):
        calls.append("note")
        return original_notes(*args, **kwargs)

    def spy_conversations(*args, **kwargs):
        calls.append("conversation")
        return original_conversations(*args, **kwargs)

    monkeypatch.setattr(service, "_media_keyword_subleg", spy_media)
    monkeypatch.setattr(RAGService, "_chacha_notes_fts", staticmethod(spy_notes))
    monkeypatch.setattr(
        RAGService, "_chacha_conversations_fts", staticmethod(spy_conversations)
    )

    results = asyncio.run(
        service._keyword_search(QUERY, top_k=20, keyword_source_types={"note"})
    )

    assert calls == ["note"], (
        f"unselected sub-legs were queried anyway: {calls}"
    )
    assert set(_types(results)) == {"note"}


def test_media_only_selection_never_opens_the_chacha_database(
    mixed_corpus, monkeypatch
):
    """The ChaChaNotes connection itself is part of the cost being saved."""
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    opened: list[str] = []

    def spy_connect(self, db_path):
        opened.append(str(db_path))
        raise AssertionError("the chacha DB must not be opened for a media-only leg")

    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", spy_connect)

    results = asyncio.run(
        service._keyword_search(QUERY, top_k=20, keyword_source_types={"media"})
    )

    assert opened == []
    assert set(_types(results)) == {"media"}


# --- Interleaving is preserved among the SELECTED types ---------------------


def test_multi_type_selection_keeps_rank_fair_interleaving(mixed_corpus):
    """AC#4: a two-type selection must not regress to concatenation."""
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    results = asyncio.run(
        service._keyword_search(
            QUERY, top_k=4, keyword_source_types={"media", "note"}
        )
    )

    assert len(results) == 4
    assert _types(results) == ["media", "note", "media", "note"], (
        "the selected sub-legs were concatenated rather than interleaved: "
        f"{_types(results)}"
    )


def test_none_means_all_three_sub_legs_unchanged(mixed_corpus):
    """Backward-compat pin: every existing caller keeps today's composition.

    These are the reviewer's measured numbers (12 of each type, top_k=20 ->
    media 7, note 7, conversation 6) -- the behavior the pushdown narrows,
    not removes.
    """
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    explicit_none = asyncio.run(
        service._keyword_search(QUERY, top_k=20, keyword_source_types=None)
    )
    default = asyncio.run(service._keyword_search(QUERY, top_k=20))

    assert _counts(explicit_none) == {"media": 7, "note": 7, "conversation": 6}
    assert _types(default) == _types(explicit_none)


def test_all_three_selected_matches_the_none_default(mixed_corpus):
    """Naming every type explicitly is the same request as naming none."""
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    selected = asyncio.run(
        service._keyword_search(
            QUERY, top_k=20, keyword_source_types={"media", "note", "conversation"}
        )
    )
    default = asyncio.run(service._keyword_search(QUERY, top_k=20))

    assert _types(selected) == _types(default)


# --- Degenerate selections --------------------------------------------------


def test_empty_selection_returns_an_empty_leg_without_querying(
    mixed_corpus, monkeypatch
):
    """An empty selection is "no keyword leg", not "the whole keyword leg"."""
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    def boom(*args, **kwargs):
        raise AssertionError("an empty selection must not query anything")

    monkeypatch.setattr(service, "_media_keyword_subleg", boom)
    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", boom)

    assert (
        asyncio.run(
            service._keyword_search(QUERY, top_k=20, keyword_source_types=set())
        )
        == []
    )


def test_unknown_source_types_are_ignored_rather_than_crashing(
    mixed_corpus, debug_logs
):
    """Fail open to fewer sub-legs: an unknown name never breaks a search.

    Failing open is only defensible if it leaves a trace -- a dropped type
    silently narrows retrieval, and the debug line is the single thread
    anyone debugging "why did my notes vanish" has to pull. Asserting the
    log is what makes "ignored with a debug log" true rather than aspirational.
    """
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    results = asyncio.run(
        service._keyword_search(
            QUERY, top_k=20, keyword_source_types={"media", "prompts", "notes"}
        )
    )

    # `prompts` has no sub-leg and `notes` is the Library's PLURAL spelling,
    # not the engine's -- both are dropped, leaving the media sub-leg alone.
    assert _counts(results) == {"media": 12}

    dropped = [
        message
        for message in debug_logs
        if "unknown keyword-leg source type" in message
    ]
    assert dropped, "dropping a source type must leave a debug trace"
    # TASK-15103: the trace is event-only — requested source types are
    # caller-controlled strings, so the ADR-029 repair stopped echoing them.
    assert "'notes'" not in dropped[0] and "'prompts'" not in dropped[0], dropped[0]


def test_the_real_engine_refuses_a_selection_on_a_semantic_search(mixed_corpus):
    """The guard must be pinned on the ENGINE, not only on a double.

    `metadata_allowlist` scopes the vector leg and `keyword_source_types`
    scopes the FTS leg; each is meaningless to the other, and the engine
    raises rather than accepting a scoping request it will not honor. That
    contract was previously asserted only by the Library suite's
    `_ProfileRagService` double -- delete the real guard and that double
    keeps happily asserting a contract nothing implements.
    """
    media_path, chacha_path = mixed_corpus
    service = _make_service(media_db_path=media_path, chachanotes_db_path=chacha_path)

    with pytest.raises(ValueError, match="keyword_source_types"):
        asyncio.run(
            service.search(
                QUERY,
                top_k=5,
                search_type="semantic",
                keyword_source_types={"note"},
            )
        )

    # `None` is the default and must stay accepted on the semantic path.
    asyncio.run(
        service.search(
            QUERY, top_k=5, search_type="semantic", keyword_source_types=None
        )
    )


# --- The cache must not serve one selection's rows to another ---------------


def _cache():
    from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache

    return SimpleRAGCache(enabled=True)


@pytest.mark.parametrize(
    "equivalent",
    [
        ["note", "media"],  # list
        ("media", "note"),  # tuple, the REVERSE of the list above
        frozenset({"note", "media"}),  # frozenset
        {"note", "media"},  # set, different literal order
    ],
)
def test_one_selection_is_one_cache_key_whatever_its_iteration_order(equivalent):
    """THE HIT DIRECTION. The miss pin below is only half the property.

    `{"media","note"}` and `["note","media"]` are the same request, and set
    iteration order for strings is hash-seed dependent, so without the
    `sorted()` in `_make_key` the same selection hashes differently from run
    to run (and between a set and the list a caller happened to build). The
    result is not a wrong answer -- it is a cache that never hits again for
    any mixed selection, silently, with every miss-direction test still
    green.

    The list and tuple cases are REVERSES of each other on purpose, and
    neither is redundant: within one process a set's iteration order is
    fixed, so it can agree with at most one of them. Deleting either leaves
    a 50/50 chance (per hash seed) that dropping `sorted()` goes unnoticed.
    Verified by mutation: `sorted(...)` -> `list(...)` reds exactly one of
    the two.
    """
    cache = _cache()

    canonical = cache._make_key(
        "quokka", "hybrid", 10, None, None, {"media", "note"}
    )

    assert (
        cache._make_key("quokka", "hybrid", 10, None, None, equivalent) == canonical
    ), f"{equivalent!r} did not canonicalize to the same key"


def test_no_selection_keeps_the_legacy_key_byte_identical():
    """Backward compat: entries written before this parameter existed must
    still be findable. `None` contributes NO key part, so the pre-TASK-14751
    five-argument call and today's six-argument one agree."""
    cache = _cache()

    legacy = cache._make_key("quokka", "hybrid", 10, None, None)
    explicit_none = cache._make_key("quokka", "hybrid", 10, None, None, None)

    assert legacy == explicit_none


def test_an_empty_selection_is_not_the_same_request_as_no_selection():
    """`set()` means "no keyword leg" and `None` means "all of it" -- two
    different searches returning different rows, so two different keys."""
    cache = _cache()

    assert cache._make_key("quokka", "hybrid", 10, None, None, set()) != cache._make_key(
        "quokka", "hybrid", 10, None, None, None
    )


def test_different_selections_get_different_cache_keys():
    """The miss direction, at the key level (the end-to-end pin is below)."""
    cache = _cache()

    assert cache._make_key(
        "quokka", "hybrid", 10, None, None, {"media"}
    ) != cache._make_key("quokka", "hybrid", 10, None, None, {"note"})


def test_selections_do_not_share_a_cache_entry(tmp_path):
    """Two selections, same query and top_k: the second must not be served
    the first's rows. The cache key is part of "the selection reaches the
    leg" -- a shared key silently reinstates the discarded-rows defect for
    every query after the first."""
    media_path = _seed_media(tmp_path, 12)
    chacha_path = _seed_chacha(tmp_path, notes=12, conversations=12)
    service = _make_service(
        media_db_path=media_path, chachanotes_db_path=chacha_path, enable_cache=True
    )

    media_only = asyncio.run(
        service._keyword_search(QUERY, top_k=20, keyword_source_types={"media"})
    )
    assert set(_types(media_only)) == {"media"}

    first = asyncio.run(
        service.search(
            QUERY, top_k=20, search_type="keyword", keyword_source_types={"media"}
        )
    )
    second = asyncio.run(
        service.search(
            QUERY, top_k=20, search_type="keyword", keyword_source_types={"note"}
        )
    )

    assert set(_types(first)) == {"media"}
    assert set(_types(second)) == {"note"}, (
        "the note-only search was served the media-only search's cached rows: "
        f"{_counts(second)}"
    )


# --- The Library's translation ---------------------------------------------


class _HybridSpyRagService:
    """`RAGService.search`'s signature plus the hybrid profile surface.

    Mirrors `Tests/Library/test_library_rag_mode_resolution._ProfileRagService`;
    the point here is only what lands in `keyword_source_types`.
    """

    def __init__(self, results=None):
        self.config = SimpleNamespace(
            search=SimpleNamespace(default_search_mode="hybrid")
        )
        self.profile = None
        self.results = results if results is not None else []
        self.calls: list[dict] = []
        self.vector_store = SimpleNamespace(
            get_collection_stats=lambda: {"count": 12}
        )

    async def search(
        self,
        query,
        top_k=None,
        search_type="semantic",
        filter_metadata=None,
        include_citations=None,
        score_threshold=None,
        metadata_allowlist=None,
        keyword_source_types=None,
    ):
        self.calls.append(
            {
                "search_type": search_type,
                "keyword_source_types": keyword_source_types,
            }
        )
        return self.results


def _note_row():
    return {
        "id": "note-1-chunk",
        "score": 0.9,
        "document": "Note evidence.",
        "metadata": {
            "title": "Note doc",
            "source_id": "note-1",
            "source_type": "note",
        },
    }


@pytest.mark.asyncio
async def test_library_hybrid_passes_the_translated_selection():
    """THE VOCABULARY PIN: the engine speaks SINGULAR source types.

    The Library's scope identifiers are plural (`notes`, `conversations`);
    the engine's keyword leg stamps and selects on the ingestion vocabulary
    (`note`, `conversation`). Handing the leg the plural spellings would
    silently select nothing at all -- every value dropped as unknown -- and
    an empty selection means an empty keyword leg.
    """
    rag = _HybridSpyRagService(results=[_note_row()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    await service.search(
        "credential", ("notes",), "rag", top_k=5, include_citations=True
    )

    assert [call["search_type"] for call in rag.calls] == ["hybrid"]
    assert rag.calls[0]["keyword_source_types"] == {"note"}


def test_every_hybrid_routable_source_type_has_a_translation():
    """The translation map's domain must cover every type that routes here.

    `_search_rag` sends a query to hybrid as soon as ONE selected type is
    FTS-servable. A servable type with no entry in the translation map would
    be dropped from `keyword_source_types` while still being routed to
    hybrid -- so the leg would quietly serve a narrower selection than the
    user picked, with no error anywhere. Also pins the reverse: a
    translation for a type that never routes here would be dead.
    """
    from tldw_chatbook.Library.library_local_rag_search_service import (
        _ENGINE_KEYWORD_SOURCE_TYPES,
        _FTS_SERVABLE_SOURCE_TYPES,
    )
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        KEYWORD_LEG_SOURCE_TYPES,
    )

    assert set(_ENGINE_KEYWORD_SOURCE_TYPES) == set(_FTS_SERVABLE_SOURCE_TYPES)
    # And every translated value must be a vocabulary the engine accepts,
    # or the pushdown would silently drop it as unknown.
    assert set(_ENGINE_KEYWORD_SOURCE_TYPES.values()) <= KEYWORD_LEG_SOURCE_TYPES


@pytest.mark.asyncio
async def test_library_hybrid_translates_every_fts_servable_type():
    """All FOUR plural identifiers translate to the engine's singular ones.

    DISCLOSED UPDATE (2026-08-11, TASK-15020/B2): this read "all three
    plural identifiers translate; `prompts` has no leg at all" until the
    prompts keyword sub-leg landed. The Search canvas's whole toggle set now
    maps onto engine sub-legs, so a selection can no longer lose a type
    between the Library and the leg.
    """
    rag = _HybridSpyRagService(results=[_note_row()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    await service.search(
        "credential",
        ("notes", "media", "conversations", "prompts"),
        "rag",
        top_k=5,
        include_citations=True,
    )

    assert rag.calls[0]["keyword_source_types"] == {
        "note",
        "media",
        "conversation",
        "prompt",
    }
