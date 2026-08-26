"""A scope allowlist must reach the FTS legs, not just the vector one.

TASK-15020 / TASK-4110, spec section "B1 -- Scope-aware hybrid". The engine
refused ``metadata_allowlist`` for anything but ``search_type="semantic"``
(one ``ValueError`` in ``RAGService.search``), so every scoped query in the
app was diverted to a semantic-only search: the keyword leg -- the only leg
that finds an exact term the embedding model has no room for -- was
structurally unreachable under a scope. The P2ab scoped fixture category
exists to measure exactly that hole, and it cannot score a keyword-found
in-scope document until the allowlist reaches the FTS sub-legs.

What this module pins:

* **Both legs.** Hybrid + allowlist runs the semantic leg exactly as before
  (per-entry store queries, merged by score) AND an FTS leg whose sub-legs
  each carry their entry's ids as ``id IN (SELECT value FROM json_each(?))``.
* **Fail-closed.** A sub-leg whose source type has NO entry in the allowlist
  is SKIPPED, never run unfiltered. That is the semantic side's semantics
  (a store query AND-scoped on ``source_type`` returns nothing for the types
  it does not name), and the difference between the two is invisible in the
  row set when the corpus happens to be single-type -- only a spy can tell
  "skipped" from "ran and found nothing", which is why the spies are here.
* **Keyword mode still refuses.** ``search_type="keyword"`` + allowlist keeps
  raising (declared spec non-goal: plain-profile scoped search uses the
  four-seam Library path, which is already scope-aware). Widening the guard
  removal past hybrid would be unmeasured scope creep.
* **Cache keys compose.** Same query + different allowlists must never share
  an entry (that is how a scoped search silently serves an unscoped one's
  rows), and the same allowlist in a different entry/key/value order must
  share one (or the cache never hits again for any scoped query).
* **json_each discipline.** A 1500-id allowlist binds ONE parameter, not
  1500 -- the SQLite variable cap is a real ceiling for real scopes, and the
  precedent (``ChaChaNotes_DB.search_notes``' ``id_allowlist``,
  ``Client_Media_DB_v2``'s ``media_ids_filter``) is json_each.

Real databases throughout (the ``test_keyword_leg_pushdown.py`` pattern).
Doubles are what let the original defect live: the Library's unit tests drive
a fake whose only allowlist behavior is the ValueError it copies.

Mutation-verified (both reds observed before commit):

* fail-closed -> unfiltered (an absent entry means "no id filter" instead of
  "skip"): ``test_fail_closed_skips_the_sub_legs_the_scope_does_not_name``
  and ``test_out_of_scope_notes_are_absent_from_a_scoped_hybrid_search`` red.
* the json_each filter dropped from ONE sub-leg (notes): the notes
  out-of-scope test reds while the media/conversation ones stay green.
"""
import asyncio
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache

# The token every seeded document carries, so the FTS leg matches all of them
# and only the allowlist can narrow the result.
QUERY = "quokka"


# --- Fixtures: real databases -----------------------------------------------


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
    """A real MediaDatabase with `count` matching articles. Returns ids."""
    db_path = tmp_path / name
    db = MediaDatabase(db_path=str(db_path), client_id="test_allowlist")
    ids = []
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
            ids.append(media_id)
    finally:
        db.close_connection()
    return db_path, ids


def _seed_chacha(tmp_path, notes, conversations, name="chacha.db"):
    """A real ChaChaNotes DB. Returns (path, note_ids, conversation_ids)."""
    db = CharactersRAGDB(tmp_path / name, "test_allowlist")
    note_ids, conv_ids = [], []
    try:
        for i in range(notes):
            note_id = db.add_note(
                f"Quokka Note {i:02d}",
                f"Field note {i}: quokka smiles are an artifact of jaw shape.",
            )
            assert note_id, "note seed failed"
            note_ids.append(note_id)
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
            conv_ids.append(conv_id)
    finally:
        db.close_connection()
    return tmp_path / name, note_ids, conv_ids


@pytest.fixture
def corpus(tmp_path):
    """Three documents of each type, in a real media DB and a real chacha DB."""
    media_path, media_ids = _seed_media(tmp_path, 3)
    chacha_path, note_ids, conv_ids = _seed_chacha(tmp_path, notes=3, conversations=3)
    return {
        "media_path": media_path,
        "chacha_path": chacha_path,
        "media": [str(i) for i in media_ids],
        "note": [str(i) for i in note_ids],
        "conversation": [str(i) for i in conv_ids],
    }


def _service_for(corpus, **kwargs):
    return _make_service(
        media_db_path=corpus["media_path"],
        chachanotes_db_path=corpus["chacha_path"],
        **kwargs,
    )


def _entry(source_type, ids):
    """The shape `rag_scope.build_semantic_allowlists` produces per type."""
    return {"source_type": {source_type}, "source_id": set(ids)}


def _source_ids(results, source_type=None):
    return [
        r.metadata.get("source_id")
        for r in results
        if source_type is None or r.metadata.get("source_type") == source_type
    ]


def _types(results):
    return [r.metadata.get("source_type") for r in results]


@pytest.fixture
def warnings_captured():
    """Collect loguru WARNING+ records (`test_keyword_leg_chacha.py`'s idiom).

    `capsys` never sees loguru output, so "we warned about it" is
    unverifiable without a sink.
    """
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# --- The guard: hybrid is admitted, keyword still refuses -------------------


def test_hybrid_with_an_allowlist_no_longer_raises(corpus):
    """The whole point of B1: a scoped hybrid search is a legal request."""
    service = _service_for(corpus)

    results = asyncio.run(
        service.search(
            QUERY,
            top_k=5,
            search_type="hybrid",
            metadata_allowlist=_entry("media", corpus["media"][:1]),
        )
    )

    assert _source_ids(results) == [corpus["media"][0]]


def test_keyword_mode_with_an_allowlist_still_raises(corpus):
    """Declared non-goal, pinned so it cannot drift open unnoticed.

    Plain-profile scoped search runs through the Library's four-seam path
    (already scope-aware, id-filtered at each DB seam); the engine's keyword
    mode has no semantic leg to scope and no caller that needs one. An
    accidental widening here would ship an unmeasured second scoped path.
    """
    service = _service_for(corpus)

    with pytest.raises(ValueError, match="metadata_allowlist"):
        asyncio.run(
            service.search(
                QUERY,
                top_k=5,
                search_type="keyword",
                metadata_allowlist=_entry("media", corpus["media"][:1]),
            )
        )


def test_an_empty_allowlist_is_still_no_allowlist_for_every_mode(corpus):
    """`None`/`{}` must keep every mode byte-for-byte as it was."""
    service = _service_for(corpus)

    for search_type in ("semantic", "hybrid", "keyword"):
        asyncio.run(
            service.search(
                QUERY, top_k=3, search_type=search_type, metadata_allowlist=None
            )
        )
        asyncio.run(
            service.search(
                QUERY, top_k=3, search_type=search_type, metadata_allowlist={}
            )
        )


# --- A scope must survive being read more than once -------------------------
#
# Review finding (reproduced live): a scope is read at least three times per
# search -- the cache key, then each leg -- and nothing materialized it. A
# generator expression was drained by the cache-key pass, so both legs
# received an EMPTY allowlist, which is "no scoping request": the search
# returned ALL rows and cached them under the SCOPED key. No caller passes one
# today; Tasks 5/7 thread allowlists through new seams, which is exactly when
# a comprehension gets written as a genexp by accident.


def test_a_one_shot_allowlist_survives_the_cache_key_pass(corpus):
    """A generator allowlist must scope the search, not silently open it."""
    service = _service_for(corpus)
    in_scope = corpus["media"][1]

    results = _hybrid_keyword_rows(
        service, (entry for entry in [_entry("media", [in_scope])])
    )

    assert _source_ids(results) == [in_scope], (
        "a one-shot allowlist was consumed before the legs read it, so the "
        f"search ran UNSCOPED: {_source_ids(results)}"
    )


def test_one_shot_id_values_survive_the_cache_key_pass(corpus):
    """Same hazard one level down: the ids themselves may be a generator."""
    service = _service_for(corpus)
    in_scope = corpus["note"][0]

    results = _hybrid_keyword_rows(
        service,
        {
            "source_type": {"note"},
            "source_id": (value for value in [in_scope]),
        },
    )

    assert _source_ids(results) == [in_scope], (
        f"a one-shot source_id value left the leg unscoped: {_source_ids(results)}"
    )


def test_a_one_shot_scope_never_caches_unscoped_rows_under_its_key(corpus):
    """The cache is where a drained scope does lasting damage.

    The key is built from the scope, so an allowlist consumed during that
    pass writes the UNSCOPED row set under the SCOPED key -- and every later
    scoped search is served those rows, correctly-materialized or not.
    """
    service = _service_for(corpus, enable_cache=True)
    in_scope = corpus["media"][0]

    from_generator = asyncio.run(
        service.search(
            QUERY,
            top_k=10,
            search_type="hybrid",
            metadata_allowlist=(entry for entry in [_entry("media", [in_scope])]),
        )
    )
    # The same request, materialized: this one hits the entry the first wrote.
    from_cache = asyncio.run(
        service.search(
            QUERY,
            top_k=10,
            search_type="hybrid",
            metadata_allowlist=[_entry("media", [in_scope])],
        )
    )

    assert _source_ids(from_generator) == [in_scope]
    assert _source_ids(from_cache) == [in_scope], (
        "the scoped key held unscoped rows: " f"{_source_ids(from_cache)}"
    )


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param([{}], id="lone-empty-entry"),
        pytest.param(
            [{"source_type": {"media"}, "source_id": {"1"}}, {}], id="empty-in-union"
        ),
    ],
)
def test_an_empty_entry_in_a_union_is_rejected(corpus, malformed):
    """An AND-group that restricts nothing is a caller defect, not a scope.

    Dropped silently, the two shapes above disagree: `[{}]` normalizes to "no
    allowlist" (fail-OPEN -- every row returned) while `[{media}, {}]` reads
    as "media only". `EffectiveScope` carries only non-empty entries, so
    neither is reachable from `build_semantic_allowlists`; refusing both is
    what keeps the fail-open direction from standing silent.
    """
    service = _service_for(corpus)

    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(
            service.search(
                QUERY, top_k=5, search_type="hybrid", metadata_allowlist=malformed
            )
        )


def test_a_source_type_with_no_sub_leg_is_named_in_a_warning(
    corpus, warnings_captured
):
    """A silently empty keyword leg is the worst possible symptom.

    Two ways to land here, both real: the Library's PLURAL spelling
    (`notes`), and a source type that reaches the scope vocabulary before it
    has an FTS sub-leg. Both sibling paths
    (`_resolve_keyword_source_types`, the Library's translation map) name
    what they dropped; this one must too, or "why did my notes vanish" has
    no thread to pull.

    DISCLOSED UPDATE (2026-08-11, TASK-15020/B2): the second example used to
    be `prompt`, since prompts reached the scope vocabulary before they had
    a sub-leg. They have one now, so `character` stands in for the class.
    That swap also fixed a pass this test had started getting for the wrong
    reason: the warning's SECOND half lists the types the leg does serve, so
    once `prompt` joined that list, `"'prompt'" in named[0]` was satisfied by
    the serves-list rather than by the dropped-list. The assertion below now
    splits the message at the serves-list boundary, so it can only be
    satisfied by the half it is about.
    """
    service = _service_for(corpus)

    results = asyncio.run(
        service._keyword_search(
            QUERY,
            top_k=10,
            metadata_allowlist={
                "source_type": {"notes", "character"},
                "source_id": set(corpus["note"]),
            },
        )
    )

    assert results == []
    named = [m for m in warnings_captured if "no keyword sub-leg serves" in m]
    assert named, "an unservable scoped source type must leave a warning"
    dropped_half = named[0].split("the leg serves")[0]
    assert "'notes'" in dropped_half and "'character'" in dropped_half, named[0]


# --- Per-sub-leg id filtering, one test per source type ---------------------


def _hybrid_keyword_rows(service, allowlist, top_k=10, **kwargs):
    """Hybrid over a deliberately empty vector index: the rows ARE the FTS leg."""
    assert service.vector_store.get_collection_stats().get("count", 0) == 0, (
        "this helper only means what it says over an empty vector index"
    )
    return asyncio.run(
        service.search(
            QUERY,
            top_k=top_k,
            search_type="hybrid",
            metadata_allowlist=allowlist,
            **kwargs,
        )
    )


def test_out_of_scope_media_is_absent_from_a_scoped_hybrid_search(corpus):
    service = _service_for(corpus)
    in_scope = corpus["media"][1]

    results = _hybrid_keyword_rows(service, _entry("media", [in_scope]))

    assert _source_ids(results) == [in_scope], (
        "the media sub-leg ignored its id filter: "
        f"{_source_ids(results)} (scope named only {in_scope})"
    )


def test_out_of_scope_notes_are_absent_from_a_scoped_hybrid_search(corpus):
    service = _service_for(corpus)
    in_scope = corpus["note"][2]

    results = _hybrid_keyword_rows(service, _entry("note", [in_scope]))

    assert _source_ids(results) == [in_scope], (
        f"the notes sub-leg ignored its id filter: {_source_ids(results)}"
    )


def test_out_of_scope_conversations_are_absent_from_a_scoped_hybrid_search(corpus):
    service = _service_for(corpus)
    in_scope = corpus["conversation"][0]

    results = _hybrid_keyword_rows(service, _entry("conversation", [in_scope]))

    assert _source_ids(results) == [in_scope], (
        f"the conversation sub-leg ignored its id filter: {_source_ids(results)}"
    )


def test_each_entry_filters_its_own_sub_leg(corpus):
    """The union case: one entry per type, each carrying ITS OWN ids.

    A flat AND-ed dict cannot express "(media in A) OR (note in B)", which is
    why `build_semantic_allowlists` returns a LIST. If the legs shared one
    pooled id set, the note whose id is not in the media entry would vanish
    (or a media row would survive on a note's id).
    """
    service = _service_for(corpus)
    media_id, note_id = corpus["media"][0], corpus["note"][1]

    results = _hybrid_keyword_rows(
        service, [_entry("media", [media_id]), _entry("note", [note_id])]
    )

    assert sorted(_source_ids(results)) == sorted([media_id, note_id])
    assert set(_types(results)) == {"media", "note"}


def test_the_media_id_filter_survives_the_int_vs_str_boundary(corpus):
    """`EffectiveScope` carries ids as STRINGS; `Media.id` is an INTEGER.

    The scope side is stringly typed end to end (`build_semantic_allowlists`
    hands the vector store str ids, and the vector store compares
    `str(metadata[key])`), so the FTS side binds `str(id)` too and relies on
    SQLite applying the left operand's NUMERIC affinity to the json_each
    values -- exactly what `Client_Media_DB_v2`'s `media_ids_filter` and
    `search_notes`' `id_allowlist` already do. If that ever stopped holding,
    every scoped media search would silently return nothing, which is the
    failure mode hardest to notice (an empty result reads as "no matches").
    """
    service = _service_for(corpus)

    results = _hybrid_keyword_rows(service, _entry("media", [corpus["media"][0]]))

    assert _source_ids(results) == [corpus["media"][0]]


# --- Fail-closed: an unnamed sub-leg is skipped, not run unfiltered ---------


def test_fail_closed_skips_the_sub_legs_the_scope_does_not_name(corpus, monkeypatch):
    """A scope naming only notes must never QUERY media or conversations.

    The row set alone cannot distinguish "skipped" from "ran unfiltered and
    got filtered downstream" -- there is no downstream filter here, so an
    unfiltered media sub-leg would leak out-of-scope rows AND cost the query.
    The spies pin the mechanism, the assertion on the rows pins the effect.
    """
    service = _service_for(corpus)

    ran: list[str] = []
    original_media = service._media_keyword_subleg
    original_notes = RAGService._chacha_notes_fts
    original_conversations = RAGService._chacha_conversations_fts

    async def spy_media(*args, **kwargs):
        ran.append("media")
        return await original_media(*args, **kwargs)

    def spy_notes(*args, **kwargs):
        ran.append("note")
        return original_notes(*args, **kwargs)

    def spy_conversations(*args, **kwargs):
        ran.append("conversation")
        return original_conversations(*args, **kwargs)

    monkeypatch.setattr(service, "_media_keyword_subleg", spy_media)
    monkeypatch.setattr(RAGService, "_chacha_notes_fts", staticmethod(spy_notes))
    monkeypatch.setattr(
        RAGService, "_chacha_conversations_fts", staticmethod(spy_conversations)
    )

    results = asyncio.run(
        service._keyword_search(
            QUERY, top_k=10, metadata_allowlist=_entry("note", corpus["note"][:1])
        )
    )

    assert ran == ["note"], f"a sub-leg the scope never named was queried: {ran}"
    assert _source_ids(results) == [corpus["note"][0]]


def test_a_notes_only_scope_never_opens_the_media_database(corpus, monkeypatch):
    """The skipped sub-leg's cost includes the connection it never opens."""
    service = _service_for(corpus)

    from tldw_chatbook.RAG_Search.simplified import rag_service as rag_service_module

    def boom(*args, **kwargs):
        raise AssertionError("the media DB must not be opened for a notes-only scope")

    monkeypatch.setattr(rag_service_module, "get_connection_pool", boom)

    results = asyncio.run(
        service._keyword_search(
            QUERY, top_k=10, metadata_allowlist=_entry("note", corpus["note"][:1])
        )
    )

    assert set(_types(results)) == {"note"}


def test_a_media_only_scope_never_opens_the_chacha_database(corpus, monkeypatch):
    service = _service_for(corpus)

    def boom(self, db_path):
        raise AssertionError("the chacha DB must not be opened for a media-only scope")

    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", boom)

    results = asyncio.run(
        service._keyword_search(
            QUERY, top_k=10, metadata_allowlist=_entry("media", corpus["media"][:1])
        )
    )

    assert set(_types(results)) == {"media"}


def test_a_scope_naming_no_selected_sub_leg_degrades_the_keyword_leg(
    corpus, monkeypatch
):
    """Empty for EVERY selected sub-leg -> `[]`, through the existing path.

    The caller selected media+notes; the scope names only conversations. No
    sub-leg can run, so the leg is empty -- and hybrid degrades to its
    semantic leg exactly as it already does for an empty FTS result, without
    touching a database on the way.
    """
    service = _service_for(corpus)

    from tldw_chatbook.RAG_Search.simplified import rag_service as rag_service_module

    def boom(*args, **kwargs):
        raise AssertionError("no sub-leg can run, so nothing may be opened")

    monkeypatch.setattr(rag_service_module, "get_connection_pool", boom)
    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", boom)

    assert (
        asyncio.run(
            service._keyword_search(
                QUERY,
                top_k=10,
                keyword_source_types={"media", "note"},
                metadata_allowlist=_entry("conversation", corpus["conversation"]),
            )
        )
        == []
    )


def test_a_scope_with_zero_ids_for_its_type_returns_no_rows(corpus, monkeypatch):
    """An entry naming a type with an EMPTY id set matches nothing.

    Same answer the semantic leg gives (`str(id) in set()` is False for every
    candidate), reached without a query. Not "no filter".
    """
    service = _service_for(corpus)

    def boom(self, db_path):
        raise AssertionError("an empty id set can only match nothing")

    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", boom)

    assert (
        asyncio.run(
            service._keyword_search(
                QUERY, top_k=10, metadata_allowlist=_entry("note", [])
            )
        )
        == []
    )


def test_a_scope_the_keyword_leg_cannot_enforce_fails_closed(corpus, monkeypatch):
    """An allowlist key no sub-leg can express must not be ignored.

    The FTS legs can enforce `source_type` and `source_id`. An entry carrying
    anything else (a future metadata dimension, a caller mistake) is a
    scoping request the leg cannot honor -- honoring it partially would run
    an under-restricted query and return rows the caller asked to exclude.
    """
    service = _service_for(corpus)

    def boom(*args, **kwargs):
        raise AssertionError("an unenforceable scope must not reach a database")

    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", boom)

    assert (
        asyncio.run(
            service._keyword_search(
                QUERY,
                top_k=10,
                metadata_allowlist={
                    "source_type": {"note"},
                    "source_id": set(corpus["note"]),
                    "author": {"Tester"},
                },
            )
        )
        == []
    )


def test_the_selection_and_the_scope_intersect(corpus):
    """`keyword_source_types` and the scope are both narrowing, never widening."""
    service = _service_for(corpus)

    results = asyncio.run(
        service._keyword_search(
            QUERY,
            top_k=10,
            keyword_source_types={"media", "note"},
            metadata_allowlist=[
                _entry("note", corpus["note"][:1]),
                _entry("conversation", corpus["conversation"]),
            ],
        )
    )

    assert _source_ids(results) == [corpus["note"][0]]


def test_no_allowlist_leaves_the_keyword_leg_exactly_as_it_was(corpus):
    """Backward-compat pin: every pre-B1 caller keeps today's composition."""
    service = _service_for(corpus)

    without = asyncio.run(service._keyword_search(QUERY, top_k=9))
    explicit_none = asyncio.run(
        service._keyword_search(QUERY, top_k=9, metadata_allowlist=None)
    )

    assert _types(without) == ["media", "note", "conversation"] * 3
    assert _source_ids(explicit_none) == _source_ids(without)


# --- The semantic leg stays scoped too --------------------------------------


def _index_chunk(service, source_type, source_id, text):
    embedding = asyncio.run(service.embeddings.create_embeddings_async([text]))[0]
    service.vector_store.add(
        ids=[f"{source_type}-{source_id}-chunk"],
        embeddings=[list(embedding)],
        documents=[text],
        metadata=[{"source_id": str(source_id), "source_type": source_type}],
    )


def _index_media_chunk(service, source_id, text):
    _index_chunk(service, "media", source_id, text)


def test_the_semantic_leg_is_scoped_by_the_same_allowlist(corpus):
    """Both legs, not just the new one.

    The vector leg's scoping is what already worked; a hybrid that pushed the
    allowlist only to the FTS leg would leak out-of-scope vector rows into
    the fused result -- the same defect this task fixes, mirrored.
    """
    service = _service_for(corpus)
    in_scope, out_of_scope = corpus["media"][0], corpus["media"][1]
    _index_media_chunk(service, in_scope, f"{QUERY} in scope passage")
    _index_media_chunk(service, out_of_scope, f"{QUERY} out of scope passage")

    results = asyncio.run(
        service.search(
            QUERY,
            top_k=10,
            search_type="hybrid",
            metadata_allowlist=_entry("media", [in_scope]),
        )
    )

    assert out_of_scope not in _source_ids(results), (
        f"an out-of-scope vector row survived a scoped hybrid: {_source_ids(results)}"
    )
    assert _source_ids(results) == [in_scope]


def test_a_multi_entry_allowlist_excludes_out_of_scope_vector_rows(corpus):
    """The union case's ROW pin, not just its call-mechanics pin.

    One entry per type, each with its own ids, and a vector index holding an
    in-scope AND an out-of-scope chunk of each type. If the per-entry store
    queries were merged from one pooled (or dropped) allowlist, the two
    out-of-scope chunks would ride along -- and a union scope is precisely
    where that mistake is easiest to make.
    """
    service = _service_for(corpus)
    media_in, media_out = corpus["media"][0], corpus["media"][1]
    note_in, note_out = corpus["note"][0], corpus["note"][1]
    for source_type, source_id in (
        ("media", media_in),
        ("media", media_out),
        ("note", note_in),
        ("note", note_out),
    ):
        _index_chunk(service, source_type, source_id, f"{QUERY} passage {source_id}")

    results = asyncio.run(
        service.search(
            QUERY,
            top_k=10,
            search_type="hybrid",
            metadata_allowlist=[
                _entry("media", [media_in]),
                _entry("note", [note_in]),
            ],
        )
    )

    assert sorted(set(_source_ids(results))) == sorted([media_in, note_in]), (
        f"out-of-scope vector rows survived a union scope: {_source_ids(results)}"
    )


def test_a_multi_entry_allowlist_queries_the_store_once_per_entry(corpus):
    """The semantic leg's union convention, unchanged: one query per entry.

    Mirrors what `library_local_rag_search_service` and
    `pipeline_functions_simple` already do for scoped semantic search -- a
    single flat allowlist cannot express the union, so each entry is its own
    store query and the results merge by score.
    """
    service = _service_for(corpus)
    seen = []
    original_search = service.vector_store.search

    def spy(query_embedding, top_k=10, *, metadata_allowlist=None):
        seen.append(metadata_allowlist)
        return original_search(
            query_embedding, top_k, metadata_allowlist=metadata_allowlist
        )

    service.vector_store.search = spy

    asyncio.run(
        service.search(
            QUERY,
            top_k=5,
            search_type="hybrid",
            include_citations=False,
            metadata_allowlist=[
                _entry("media", corpus["media"][:1]),
                _entry("note", corpus["note"][:1]),
            ],
        )
    )

    assert len(seen) == 2, f"expected one store query per entry, got {seen}"
    assert {next(iter(a["source_type"])) for a in seen} == {"media", "note"}


# --- json_each discipline ---------------------------------------------------


class _RecordingConnection:
    """Proxies a real sqlite3 connection, recording every executed statement."""

    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def execute(self, sql, params=()):
        self._log.append((sql, tuple(params)))
        return self._inner.execute(sql, params)

    def close(self):
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RecordingCursor:
    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def execute(self, sql, params=()):
        self._log.append((sql, tuple(params)))
        return self._inner.execute(sql, params)

    def __iter__(self):
        return iter(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RecordingPool:
    """A connection-pool stand-in over one real sqlite3 connection."""

    def __init__(self, conn, log):
        self._conn = conn
        self._log = log

    @contextmanager
    def transaction(self):
        yield self

    def cursor(self):
        return _RecordingCursor(self._conn.cursor(), self._log)


def test_a_thousand_note_ids_bind_as_one_json_parameter(corpus, monkeypatch):
    """SQLite's variable cap is a real ceiling; json_each is the precedent.

    A per-id placeholder list would make a ~1k-item scope (an entirely
    ordinary collection) raise `OperationalError: too many SQL variables` on
    builds whose SQLITE_MAX_VARIABLE_NUMBER is 999 -- and the sub-leg swallows
    its own errors, so the symptom would be a silently empty scoped search.
    """
    service = _service_for(corpus)
    executed: list[tuple[str, tuple]] = []
    original = RAGService._connect_chacha_readonly

    def recording(self, db_path):
        return _RecordingConnection(original(self, db_path), executed)

    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", recording)

    ids = [f"absent-note-{i}" for i in range(1500)] + [corpus["note"][0]]
    results = asyncio.run(
        service._keyword_search(
            QUERY,
            top_k=10,
            keyword_source_types={"note"},
            metadata_allowlist=_entry("note", ids),
        )
    )

    assert _source_ids(results) == [corpus["note"][0]]
    notes_statements = [(sql, params) for sql, params in executed if "notes_fts" in sql]
    assert len(notes_statements) == 1, notes_statements
    sql, params = notes_statements[0]
    assert "json_each(?)" in sql, sql
    assert len(params) == 3, (
        f"the ids did not bind as ONE json parameter: {len(params)} parameters"
    )
    assert json.loads(params[1]) == sorted(ids)


def test_a_thousand_conversation_ids_bind_as_one_json_parameter(corpus, monkeypatch):
    """The third sub-leg, asserted on its own SQL rather than by analogy.

    It is the one that does NOT simply mirror the notes shape: it filters the
    conversation (its unit of retrieval) inside a grouped query, and it runs
    a second statement afterwards. Only a capture proves the filter landed on
    `c.id` in the first statement and that the second stayed bounded by the
    conversations that survived it.
    """
    service = _service_for(corpus)
    executed: list[tuple[str, tuple]] = []
    original = RAGService._connect_chacha_readonly

    def recording(self, db_path):
        return _RecordingConnection(original(self, db_path), executed)

    monkeypatch.setattr(RAGService, "_connect_chacha_readonly", recording)

    ids = [f"absent-conversation-{i}" for i in range(1500)] + [
        corpus["conversation"][1]
    ]
    results = asyncio.run(
        service._keyword_search(
            QUERY,
            top_k=10,
            keyword_source_types={"conversation"},
            metadata_allowlist=_entry("conversation", ids),
        )
    )

    assert _source_ids(results) == [corpus["conversation"][1]]
    grouped = [(sql, params) for sql, params in executed if "GROUP BY c.id" in sql]
    assert len(grouped) == 1, grouped
    sql, params = grouped[0]
    assert "c.id IN (SELECT value FROM json_each(?))" in sql, sql
    assert len(params) == 3, (
        f"the ids did not bind as ONE json parameter: {len(params)} parameters"
    )
    assert json.loads(params[1]) == sorted(ids)

    # The follow-up statement is bounded by the surviving conversations, so
    # the 1500 absent ids never reach it as placeholders either.
    messages = [(s, p) for s, p in executed if "m.conversation_id IN" in s]
    assert len(messages) == 1 and len(messages[0][1]) == 2, messages


def test_the_media_sub_leg_binds_its_ids_as_one_json_parameter(corpus):
    """Same discipline on the media sub-leg, asserted on the real SQL."""
    service = _service_for(corpus)
    conn = sqlite3.connect(str(corpus["media_path"]))
    conn.row_factory = sqlite3.Row
    executed: list[tuple[str, tuple]] = []
    pool = _RecordingPool(conn, executed)
    ids = [str(1_000_000 + i) for i in range(1500)] + [corpus["media"][0]]

    try:
        rows = service._perform_fts5_search(pool, QUERY, 10, ids)
    finally:
        conn.close()

    assert [str(row["id"]) for row in rows] == [corpus["media"][0]]
    assert len(executed) == 1
    sql, params = executed[0]
    assert "json_each(?)" in sql, sql
    assert len(params) == 3, (
        f"the ids did not bind as ONE json parameter: {len(params)} parameters"
    )


def test_no_allowlist_leaves_the_media_sql_without_an_id_filter(corpus):
    """The unscoped query shape must not grow a filter it does not need."""
    service = _service_for(corpus)
    conn = sqlite3.connect(str(corpus["media_path"]))
    conn.row_factory = sqlite3.Row
    executed: list[tuple[str, tuple]] = []

    try:
        service._perform_fts5_search(_RecordingPool(conn, executed), QUERY, 10)
    finally:
        conn.close()

    sql, params = executed[0]
    assert "json_each" not in sql, sql
    assert len(params) == 2


# --- Cache-key composition --------------------------------------------------


def _key(cache, **kwargs):
    return cache._make_key(
        kwargs.pop("query", "quokka"),
        kwargs.pop("search_type", "hybrid"),
        kwargs.pop("top_k", 10),
        kwargs.pop("filters", None),
        kwargs.pop("metadata_allowlist", None),
        kwargs.pop("keyword_source_types", None),
        kwargs.pop("hybrid_fusion", None),
    )


def test_different_allowlists_get_different_cache_keys():
    """THE MISS DIRECTION: a shared key silently serves one scope's rows to
    another -- the exact failure the whole scope feature exists to prevent."""
    cache = SimpleRAGCache(enabled=True)

    assert _key(cache, metadata_allowlist=_entry("media", ["1"])) != _key(
        cache, metadata_allowlist=_entry("media", ["2"])
    )
    assert _key(cache, metadata_allowlist=_entry("media", ["1"])) != _key(
        cache, metadata_allowlist=_entry("note", ["1"])
    )
    assert _key(
        cache, metadata_allowlist=[_entry("media", ["1"]), _entry("note", ["2"])]
    ) != _key(cache, metadata_allowlist=[_entry("media", ["1"])])


def test_one_allowlist_is_one_cache_key_whatever_its_order():
    """THE HIT DIRECTION. Set iteration order for strings is hash-seed
    dependent, and `build_semantic_allowlists` sorts by source_type while a
    hand-built list need not -- without canonicalization the same scope
    hashes differently between runs and the cache never hits again."""
    cache = SimpleRAGCache(enabled=True)

    forward = [_entry("media", ["1", "2"]), _entry("note", ["a", "b"])]
    reversed_entries = [_entry("note", ["b", "a"]), _entry("media", ["2", "1"])]
    assert _key(cache, metadata_allowlist=forward) == _key(
        cache, metadata_allowlist=reversed_entries
    )

    # Within one entry: dict key order and value iteration order.
    assert _key(
        cache, metadata_allowlist={"source_type": {"media"}, "source_id": {"1", "2"}}
    ) == _key(
        cache, metadata_allowlist={"source_id": {"2", "1"}, "source_type": {"media"}}
    )


def test_a_one_entry_list_is_the_same_request_as_the_bare_mapping():
    """`build_semantic_allowlists` returns a list even for a single type; the
    consumers that unwrap it (semantic, today) must not split the cache."""
    cache = SimpleRAGCache(enabled=True)

    assert _key(cache, metadata_allowlist=[_entry("media", ["1"])]) == _key(
        cache, metadata_allowlist=_entry("media", ["1"])
    )


def test_a_multi_entry_union_is_not_one_and_ed_entry():
    """`[{a},{b}]` is a UNION and `{a,b}` is an intersection -- two different
    searches, so two different keys. A canonicalization that flattened the
    entries would collapse them onto one."""
    cache = SimpleRAGCache(enabled=True)

    union = [{"source_type": {"media"}}, {"source_id": {"1"}}]
    intersection = {"source_type": {"media"}, "source_id": {"1"}}
    assert _key(cache, metadata_allowlist=union) != _key(
        cache, metadata_allowlist=intersection
    )


def test_the_allowlist_composes_with_the_fusion_and_selection_key_parts():
    """Each part must be independently load-bearing on a hybrid key."""
    cache = SimpleRAGCache(enabled=True)
    allowlist = _entry("media", ["1"])

    base = _key(
        cache,
        metadata_allowlist=allowlist,
        keyword_source_types={"media"},
        hybrid_fusion=(0.7, 5, 2),
    )
    other_allowlist = _key(
        cache,
        metadata_allowlist=_entry("media", ["2"]),
        keyword_source_types={"media"},
        hybrid_fusion=(0.7, 5, 2),
    )
    other_fusion = _key(
        cache,
        metadata_allowlist=allowlist,
        keyword_source_types={"media"},
        hybrid_fusion=(0.7, 60, 2),
    )
    other_selection = _key(
        cache,
        metadata_allowlist=allowlist,
        keyword_source_types={"note"},
        hybrid_fusion=(0.7, 5, 2),
    )

    assert len({base, other_allowlist, other_fusion, other_selection}) == 4


def _legacy_hash(key_str):
    """`_make_key`'s hashing tail, reproduced so "byte-identical" is checkable."""
    try:
        import xxhash

        return xxhash.xxh64(key_str.encode()).hexdigest()
    except ImportError:
        import hashlib

        return hashlib.md5(key_str.encode()).hexdigest()


def test_no_allowlist_keeps_the_legacy_key_byte_identical():
    """Backward compat, both directions: `None` contributes no key part, and
    a single-entry allowlist hashes exactly as it did before this task."""
    cache = SimpleRAGCache(enabled=True)

    assert cache._make_key("quokka", "hybrid", 10, None) == _key(cache)
    # The pre-B1 rendering of a one-entry allowlist, reproduced literally.
    canonical = frozenset(
        (k, tuple(sorted(str(x) for x in v)))
        for k, v in _entry("media", ["1", "2"]).items()
    )
    legacy_parts = [
        "quokka",
        "hybrid",
        "10",
        json.dumps({}, sort_keys=True),
        json.dumps(sorted(canonical)),
    ]
    assert _key(cache, metadata_allowlist=_entry("media", ["1", "2"])) == _legacy_hash(
        "|".join(legacy_parts)
    )


def test_two_scopes_do_not_share_a_cached_hybrid_result(corpus):
    """End to end: the second scoped search must not be served the first's rows."""
    service = _service_for(corpus, enable_cache=True)

    first = asyncio.run(
        service.search(
            QUERY,
            top_k=5,
            search_type="hybrid",
            metadata_allowlist=_entry("media", corpus["media"][:1]),
        )
    )
    second = asyncio.run(
        service.search(
            QUERY,
            top_k=5,
            search_type="hybrid",
            metadata_allowlist=_entry("note", corpus["note"][:1]),
        )
    )

    assert _source_ids(first) == [corpus["media"][0]]
    assert _source_ids(second) == [corpus["note"][0]], (
        "the note-scoped search was served the media-scoped search's rows: "
        f"{_source_ids(second)}"
    )


# --- The class the runtime actually resolves --------------------------------


def test_the_runtime_service_class_accepts_a_scoped_hybrid_search(corpus):
    """`EnhancedRAGServiceV2` overrides `search()` with an explicit signature.

    TASK-14751's lesson, verbatim: the P1 eval harness drives THIS class, and
    a base-class kwarg it does not forward crashes with a `TypeError` while
    every double-driven unit test stays green. The scoped harness runs land
    here, so the LIST shape has to survive this override too.
    """
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
        EnhancedRAGServiceV2,
    )

    service = EnhancedRAGServiceV2(config=_service_for(corpus).config)

    results = asyncio.run(
        service.search(
            QUERY,
            top_k=10,
            search_type="hybrid",
            metadata_allowlist=[
                _entry("media", corpus["media"][:1]),
                _entry("note", corpus["note"][:1]),
            ],
        )
    )

    assert sorted(_source_ids(results)) == sorted(
        [corpus["media"][0], corpus["note"][0]]
    )
