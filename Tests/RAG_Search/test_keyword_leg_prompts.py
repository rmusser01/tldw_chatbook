"""The engine's FTS leg must cover saved prompts too (TASK-15020 / B2).

`test_keyword_leg_chacha.py` is this file's template, and the two suites hold
down the same properties for different databases. What makes the prompts
sub-leg different -- and worth its own file -- is that **prompts have no
vector index anywhere**. Media, notes and conversations all reach hybrid
results through two legs; a prompt row can only ever arrive through this one,
and it survives fusion only because TASK-3994's weighting gives an FTS-only
row a rescue path (rrf_k=5). So the end-to-end pin here is not "a prompt row
exists" but "a prompt row, with `vector_rank=None`, lands in the fused
top-k".

Four properties, all deliberate:

* **Read-only, no ORM.** The engine opens the Prompts database through
  `connect_private_sqlite("rag.prompts_keyword_leg", ..., read_only=True)`,
  so it is structurally incapable of writing and never runs
  `PromptsDatabase`'s constructor-time schema/migration work. Tests may use
  the ORM to *build* fixtures; the engine may not.
* **The ORM's own filters, replicated.** `Prompts_DB.search_prompts` /
  `search_prompts_by_text` both resolve `prompts_fts` rowids and then join
  `Prompts` with `deleted = 0`. The raw sub-leg replicates that predicate,
  and `test_deleted_prompts_are_excluded` rebuilds the FTS index first --
  without the rebuild the predicate is untestable (soft delete evicts the
  row from the index, so dropping the line changes nothing), which is the
  vacuous-guard lesson `test_keyword_leg_chacha` learned the hard way.
* **Fusion-key vocabulary equality.** `_fusion_doc_key` compares raw
  `source_type` strings. Prompt rows stamp the SINGULAR `prompt`, matching
  the Library's `_prompt_row` provenance and `canonicalize.SOURCE_TYPE_
  ALIASES`; a plural spelling would leave prompt rows present but unable to
  merge or be post-filtered.
* **Fail-closed under a scope.** The scope vocabulary is media/note only
  (spec D5), so no allowlist can ever NAME prompts. A scoped search
  therefore skips the prompts sub-leg entirely rather than running it
  unfiltered -- pinned here, because "skip" and "run unfiltered" look
  identical on a corpus with one prompt.
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

CLIENT_ID = "test_prompts_subleg"


def _make_service(media_db_path=None, chachanotes_db_path=None, prompts_db_path=None):
    """A RAGService with the in-memory vector store and mock embeddings."""
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    if chachanotes_db_path is not None:
        cfg.search.chachanotes_db_path = Path(chachanotes_db_path)
    if prompts_db_path is not None:
        cfg.search.prompts_db_path = Path(prompts_db_path)
    return RAGService(cfg)


def _seed_prompts(tmp_path, rows, name="prompts.db"):
    """Create a real PromptsDatabase (its writer maintains `prompts_fts`).

    Args:
        tmp_path: Scratch directory.
        rows: ``(name, system_prompt)`` pairs.
        name: Database filename.

    Returns:
        ``(db_path, {name: prompt_id})``.
    """
    db_path = tmp_path / name
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    ids = {}
    try:
        for prompt_name, system_prompt in rows:
            prompt_id, _uuid, message = db.add_prompt(
                name=prompt_name,
                author=None,
                details=None,
                system_prompt=system_prompt,
            )
            assert prompt_id is not None, f"prompt seed failed: {message}"
            ids[prompt_name] = prompt_id
    finally:
        db.close_connection()
    return db_path, ids


def _seed_media(tmp_path, rows, name="tldw_cli_media_v2.db"):
    db_path = tmp_path / name
    db = MediaDatabase(db_path=str(db_path), client_id=CLIENT_ID)
    try:
        for title, content in rows:
            media_id, _, message = db.add_media_with_keywords(
                title=title,
                content=content,
                media_type="article",
                author="Tester",
                url=f"https://example.com/{title.replace(' ', '-').lower()}",
            )
            assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _rows_by_type(results):
    out = {}
    for result in results:
        out.setdefault(result.metadata.get("source_type"), []).append(result)
    return out


@pytest.fixture
def warnings_captured():
    """Collect loguru WARNING+ records (pytest's capsys never sees loguru)."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_keyword_leg_returns_prompt_rows(tmp_path):
    """A prompt is reachable through the keyword leg, stamped `prompt`."""
    db_path, ids = _seed_prompts(
        tmp_path,
        [
            (
                "Wombat shift handover",
                "Summarise the wombat burrow inspection log for the "
                "incoming supervisor.",
            )
        ],
    )

    service = _make_service(prompts_db_path=db_path)
    results = asyncio.run(service._keyword_search("wombat", top_k=10))

    rows = _rows_by_type(results).get("prompt") or []
    assert rows, f"the prompts sub-leg returned nothing: {_rows_by_type(results)}"
    row = rows[0]
    prompt_id = ids["Wombat shift handover"]
    assert row.metadata["source_id"] == str(prompt_id)
    assert row.metadata["doc_id"] == str(prompt_id)
    assert row.metadata["title"] == "Wombat shift handover"
    assert row.id == f"prompt_{prompt_id}"
    assert "burrow inspection log" in row.document, row.document


def test_prompt_body_columns_reach_the_row_document(tmp_path):
    """The row's text is the prompt's own body, not just its name.

    `prompts_fts` indexes five columns; `name`/`author` are this row's
    metadata (title/author) and the three body columns are its document.
    A sub-leg that returned only the name would match on a body term and
    then show the user nothing containing it.
    """
    db_path = tmp_path / "prompts.db"
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    try:
        prompt_id, _uuid, message = db.add_prompt(
            name="Numbat review",
            author="Numbat Author",
            details="Numbat details line.",
            system_prompt="Numbat system instruction.",
            user_prompt="Numbat user instruction.",
        )
        assert prompt_id is not None, message
    finally:
        db.close_connection()

    service = _make_service(prompts_db_path=db_path)
    results = asyncio.run(
        service._keyword_search("numbat", top_k=5, include_citations=False)
    )

    rows = _rows_by_type(results).get("prompt") or []
    assert rows, "the prompts sub-leg returned nothing"
    document = rows[0].document
    for fragment in (
        "Numbat details line.",
        "Numbat system instruction.",
        "Numbat user instruction.",
    ):
        assert fragment in document, f"{fragment!r} missing from {document!r}"
    assert rows[0].metadata["author"] == "Numbat Author"


def test_sub_legs_interleave_rank_fairly_with_prompts(tmp_path):
    """Four sub-legs now share the budget; none may crowd the others out."""
    media_path = _seed_media(
        tmp_path,
        [
            (f"Pelican Media {i}", f"Media document {i} discussing pelican migration.")
            for i in range(5)
        ],
    )
    chacha = CharactersRAGDB(tmp_path / "chacha.db", CLIENT_ID)
    chacha.add_note("Pelican Note", "Pelican pouches hold more than their stomachs.")
    conv_id = chacha.add_conversation({"title": "Pelican Chat"})
    chacha.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "Do pelican flocks migrate at night?",
        }
    )
    chacha.close_connection()
    prompts_path, _ = _seed_prompts(
        tmp_path,
        [("Pelican Prompt", "Write a pelican migration briefing for the warden.")],
    )

    service = _make_service(
        media_db_path=media_path,
        chachanotes_db_path=tmp_path / "chacha.db",
        prompts_db_path=prompts_path,
    )
    results = asyncio.run(service._keyword_search("pelican", top_k=4))

    assert len(results) == 4
    assert {r.metadata.get("source_type") for r in results} == {
        "media",
        "note",
        "conversation",
        "prompt",
    }, (
        "top_k=4 was not shared rank-fairly across the four sub-legs: "
        f"{[r.metadata.get('source_type') for r in results]}"
    )


def test_deleted_prompts_are_excluded(tmp_path):
    """A soft-deleted prompt must never leak out of the raw sub-leg.

    `Prompts_DB` maintains `prompts_fts` by hand (`_delete_fts_prompt`
    removes the row on soft delete), so the `deleted = 0` predicate the ORM
    applies looks redundant -- and is not. An external-content `'rebuild'`
    re-indexes the CONTENT TABLE, soft-deleted rows included, and the
    predicate is then the only thing keeping a deleted prompt out of search
    results. The rebuild below is what makes this test able to fail at all;
    without it, dropping the predicate changes nothing.
    """
    db_path = tmp_path / "prompts.db"
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    try:
        live_id, _uuid, _msg = db.add_prompt(
            name="Live quokka prompt",
            author=None,
            details=None,
            system_prompt="Summarise quokka sightings on the island.",
        )
        dead_id, _uuid, _msg = db.add_prompt(
            name="Dead quokka prompt",
            author=None,
            details=None,
            system_prompt="Quokka prompt slated for deletion.",
        )
        assert db.soft_delete_prompt(dead_id)
        # Reproduce a post-maintenance index: rebuilt from the content
        # table, soft-deleted rows and all.
        with db.transaction() as conn:
            conn.execute("INSERT INTO prompts_fts(prompts_fts) VALUES('rebuild')")
    finally:
        db.close_connection()

    service = _make_service(prompts_db_path=db_path)
    results = asyncio.run(service._keyword_search("quokka", top_k=10))

    ids = {r.metadata.get("source_id") for r in results}
    assert str(live_id) in ids
    assert str(dead_id) not in ids, "soft-deleted prompt leaked into the FTS leg"


def test_missing_prompts_db_degrades_to_the_other_sub_legs_with_a_warning(
    tmp_path, warnings_captured
):
    """A missing Prompts DB costs its sub-leg, not the whole leg."""
    media_path = _seed_media(
        tmp_path, [("Numbat Media", "Media coverage of numbat foraging habits.")]
    )

    service = _make_service(
        media_db_path=media_path, prompts_db_path=tmp_path / "absent-prompts.db"
    )
    results = asyncio.run(service._keyword_search("numbat", top_k=10))

    assert results, "the media sub-leg must keep working when prompts is missing"
    assert {r.metadata.get("source_type") for r in results} == {"media"}
    assert not (tmp_path / "absent-prompts.db").exists(), (
        "search must never create a database"
    )
    assert any(
        "Prompts database not found" in message for message in warnings_captured
    ), f"a missing prompts DB degraded silently; warnings: {warnings_captured}"
    assert all("absent-prompts.db" not in message for message in warnings_captured)


def test_prompts_connection_is_read_only(tmp_path):
    """The engine's prompts connection cannot write, by construction."""
    db_path, _ = _seed_prompts(
        tmp_path, [("Read Only Probe", "Content used to prove the connection reads.")]
    )

    service = _make_service(prompts_db_path=db_path)
    conn = service._connect_prompts_readonly(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM Prompts").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO Prompts (name, uuid, client_id, version) "
                "VALUES ('probe', 'x', 'engine', 1)"
            )
    finally:
        conn.close()


def test_prompts_connection_reads_a_live_wal_database(tmp_path):
    """The read-only leg must work against a DB the app is holding open."""
    db_path = tmp_path / "prompts.db"
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    try:
        prompt_id, _uuid, message = db.add_prompt(
            name="Wallaby prompt",
            author=None,
            details=None,
            system_prompt="Record wallaby sightings in the scrub.",
        )
        assert prompt_id is not None, message
        assert db.execute_query("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert (tmp_path / "prompts.db-wal").exists(), "no WAL sidecar to read across"
        # deliberately NOT closed: the writer still holds the database
        service = _make_service(prompts_db_path=db_path)
        results = asyncio.run(service._keyword_search("wallaby", top_k=5))
        assert [r.metadata["source_id"] for r in results] == [str(prompt_id)]
    finally:
        db.close_connection()


def test_traversal_shaped_prompts_path_is_rejected_before_any_db_open(
    tmp_path, warnings_captured
):
    """`prompts_db_path` gets `media_db_path`'s path_validation treatment."""
    real_subdir = tmp_path / "a" / "b"
    real_subdir.mkdir(parents=True)
    _seed_prompts(
        tmp_path, [("Traversal Bait", "Numbat prompt reachable only via traversal.")]
    )

    malicious_path = str(real_subdir / ".." / ".." / "prompts.db")
    assert "../.." in malicious_path, "test setup must produce a traversal string"

    service = _make_service(prompts_db_path=malicious_path)
    results = asyncio.run(service._keyword_search("numbat", top_k=5))

    assert results == []
    assert any("prompts_db_path" in m for m in warnings_captured), (
        f"rejection was silent; warnings: {warnings_captured}"
    )


@pytest.mark.parametrize("link_kind", ["file", "parent_dir"])
def test_symlinked_prompts_path_yields_empty_and_no_db_read(
    tmp_path, link_kind, warnings_captured
):
    """Neither a symlinked DB file nor a symlinked PARENT may be followed.

    The refusal comes from the private-SQLite seam's per-component
    `O_NOFOLLOW` walk, not from a hand-rolled final-component check (which
    the chacha sub-leg's review proved strictly weaker).
    """
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    _seed_prompts(
        outside_dir,
        [("Symlink Bait", "Bandicoot prompt that must stay unreachable.")],
        name="real_prompts.db",
    )

    if link_kind == "file":
        link = tmp_path / "prompts_via_symlink.db"
        link.symlink_to(outside_dir / "real_prompts.db")
        configured = link
    else:
        link_dir = tmp_path / "dir_via_symlink"
        link_dir.symlink_to(outside_dir, target_is_directory=True)
        configured = link_dir / "real_prompts.db"

    service = _make_service(prompts_db_path=configured)
    results = asyncio.run(service._keyword_search("bandicoot", top_k=5))

    assert results == [], f"a symlinked {link_kind} was followed to real data"
    assert warnings_captured, "a refused prompts path must be disclosed, not silent"


def test_unopenable_prompts_file_degrades_with_a_warning(tmp_path, warnings_captured):
    """A file that exists but is not a database costs only its sub-leg."""
    media_path = _seed_media(
        tmp_path, [("Bilby Media", "Media coverage of bilby burrows.")]
    )
    junk = tmp_path / "prompts.db"
    junk.write_bytes(b"this is not a SQLite database")

    service = _make_service(media_db_path=media_path, prompts_db_path=junk)
    results = asyncio.run(service._keyword_search("bilby", top_k=5))

    assert {r.metadata.get("source_type") for r in results} == {"media"}, (
        "an unopenable prompts DB must not take the media sub-leg down with it"
    )
    assert warnings_captured, "an unusable prompts DB must be disclosed"


def test_prompt_is_in_every_vocabulary_the_leg_crosses():
    """One spelling, four maps. A drift here is silent by construction.

    `prompt` has to be simultaneously: a keyword-leg source type (or the
    sub-leg is never selected), a Library FTS-servable scope identifier (or
    a prompts-only selection is diverted to semantic), a translated engine
    type (or the pushdown drops it as unknown), and a canonicalizable
    provenance value (or its rows survive every source-type post-filter,
    including one that deselected Prompts).
    """
    from tldw_chatbook.Library.library_local_rag_search_service import (
        _ENGINE_KEYWORD_SOURCE_TYPES,
        _FTS_SERVABLE_SOURCE_TYPES,
        _SEMANTIC_SOURCE_TYPE_MAP,
        _SEMANTICALLY_COVERABLE_SOURCE_TYPES,
    )
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        KEYWORD_LEG_SOURCE_TYPES,
        SOURCE_TYPE_PROMPT,
    )

    assert SOURCE_TYPE_PROMPT == "prompt"
    assert SOURCE_TYPE_PROMPT in KEYWORD_LEG_SOURCE_TYPES
    assert "prompts" in _FTS_SERVABLE_SOURCE_TYPES
    assert _ENGINE_KEYWORD_SOURCE_TYPES["prompts"] == SOURCE_TYPE_PROMPT
    assert _SEMANTIC_SOURCE_TYPE_MAP[SOURCE_TYPE_PROMPT] == "prompts"
    # ...but prompts are still NOT semantically coverable: there is no
    # vector index for them, so listing them in the coverage partition
    # would nag "semantic search found nothing from prompts" forever
    # (task-15 finding I2). This set stopped being derivable from the
    # canonicalization map's values the moment `prompt` joined it.
    assert "prompts" not in _SEMANTICALLY_COVERABLE_SOURCE_TYPES


def test_a_prompts_only_selection_gets_the_whole_keyword_budget(tmp_path):
    """Single-type selection pushdown reaches the prompts sub-leg too."""
    media_path = _seed_media(
        tmp_path,
        [(f"Bilby Media {i}", f"Media {i} about bilby burrows.") for i in range(3)],
    )
    prompts_path, _ = _seed_prompts(
        tmp_path,
        [
            (f"Bilby Prompt {i}", f"Draft bilby burrow note number {i}.")
            for i in range(4)
        ],
    )

    service = _make_service(media_db_path=media_path, prompts_db_path=prompts_path)
    results = asyncio.run(
        service._keyword_search(
            "bilby", top_k=4, keyword_source_types={"prompt"}
        )
    )

    assert len(results) == 4
    assert {r.metadata.get("source_type") for r in results} == {"prompt"}, (
        "a prompts-only selection must spend the whole budget on prompts"
    )


def test_a_scope_skips_the_prompts_sub_leg_entirely(tmp_path):
    """Fail-closed: no allowlist can name prompts, so a scope excludes them.

    The retrieval scope's vocabulary is media/note only (spec D5) -- there
    is no shape in which `build_semantic_allowlists` emits a `prompt` entry.
    A sub-leg the allowlist never names is SKIPPED rather than run
    unfiltered, so a scoped search can never surface an out-of-scope prompt.
    The same query without a scope DOES return the prompt, which is what
    makes this a pin on the scope rather than on the corpus.
    """
    prompts_path, ids = _seed_prompts(
        tmp_path, [("Scoped Bandicoot Prompt", "Bandicoot burrow briefing.")]
    )
    chacha = CharactersRAGDB(tmp_path / "chacha.db", CLIENT_ID)
    note_id = chacha.add_note("Bandicoot Note", "Bandicoot burrow observations.")
    chacha.close_connection()

    service = _make_service(
        chachanotes_db_path=tmp_path / "chacha.db", prompts_db_path=prompts_path
    )

    unscoped = asyncio.run(service._keyword_search("bandicoot", top_k=10))
    assert "prompt" in _rows_by_type(unscoped), (
        "without a scope the prompt must be reachable, or the scoped "
        "assertion below proves nothing"
    )

    scoped = asyncio.run(
        service._keyword_search(
            "bandicoot",
            top_k=10,
            metadata_allowlist=[{"source_type": {"note"}, "source_id": {str(note_id)}}],
        )
    )
    by_type = _rows_by_type(scoped)
    assert "prompt" not in by_type, (
        "a scoped search returned a prompt row; the sub-leg ran unfiltered "
        f"instead of being skipped: {by_type}"
    )
    assert [r.metadata["source_id"] for r in scoped] == [str(note_id)]
    assert str(ids["Scoped Bandicoot Prompt"]) not in {
        r.metadata["source_id"] for r in scoped
    }


def test_a_prompt_reaches_the_fused_top_k_as_an_fts_only_row(tmp_path):
    """THE rescue pin: prompts have no vector leg, so fusion is their gate.

    Every other source type reaches hybrid results two ways. A prompt row
    carries `vector_rank is None` by construction, so it is in the fused
    output only because TASK-3994's weighting scores an FTS-only row
    `(1 - alpha) / (rrf_k + fts_rank)` and this build's `rrf_k` (5, see
    `config.DEFAULT_HYBRID_RRF_K`) makes that enough to displace the tail of
    a full semantic leg.

    Fusion runs with the SERVICE'S OWN alpha and rrf_k, read off the config
    the pipeline reads them off -- not with `_fuse_hybrid_results`' bare
    defaults, whose `rrf_k` is the server-parity 60 that no shipped search
    path uses. The counterfactual at the bottom is that same 60, run rather
    than argued: it is what "the rescue is the mechanism" means, and this
    arc has twice shipped a fusion claim that paper arithmetic supported and
    the engine refuted.
    """
    from tldw_chatbook.RAG_Search.fusion import (
        resolve_hybrid_alpha,
        resolve_rrf_k,
    )

    prompts_path, ids = _seed_prompts(
        tmp_path, [("Quoll Handover Prompt", "Summarise the quoll survey handover.")]
    )
    service = _make_service(prompts_db_path=prompts_path)
    alpha = resolve_hybrid_alpha(service.config.search.hybrid_alpha)
    rrf_k = resolve_rrf_k(service.config.search.rrf_k)

    keyword_results = asyncio.run(
        service._keyword_search("quoll", top_k=10, include_citations=False)
    )
    assert keyword_results, "the prompts sub-leg produced no row to fuse"

    # A full semantic leg of unrelated documents: the prompt row has to earn
    # its slot against them, not fall into an empty result set.
    semantic_results = [
        SearchResult(
            id=f"media_{index}_chunk_0",
            score=0.9 - index * 0.01,
            document=f"Unrelated media chunk {index}.",
            metadata={
                "source_type": "media",
                "source_id": str(index),
                "doc_id": f"media_{index}",
                "chunk_id": f"media_{index}_chunk_0",
                "chunk_index": 0,
            },
        )
        for index in range(1, 11)
    ]

    def _fuse(k):
        return RAGService._fuse_hybrid_results(
            keyword_results=keyword_results,
            semantic_results=semantic_results,
            top_k=10,
            alpha=alpha,
            rrf_k=k,
            include_citations=False,
        )

    fused = _fuse(rrf_k)
    prompt_rows = [
        row for row in fused if row.metadata.get("source_type") == "prompt"
    ]
    assert prompt_rows, (
        "the prompt row did not survive fusion into the top-10; prompts have "
        "no other path into hybrid results"
    )
    provenance = prompt_rows[0].metadata["hybrid_fusion"]
    assert provenance["vector_rank"] is None, (
        "a prompt row with a vector rank means something is indexing prompts "
        f"semantically: {provenance}"
    )
    assert provenance["vector_score"] is None
    assert provenance["fts_rank"] == 1
    assert str(ids["Quoll Handover Prompt"]) == str(
        prompt_rows[0].metadata["source_id"]
    )

    # The counterfactual, measured: at the server's rrf_k=60 the same row,
    # from the same legs, does NOT make the top-10. B2's reachability and
    # TASK-3994's weighting are one fact, not two.
    starved = _fuse(60)
    assert not [
        row for row in starved if row.metadata.get("source_type") == "prompt"
    ], (
        "the prompt row survived at rrf_k=60 too, so this test is no longer "
        "measuring the rescue -- re-derive the mechanism before trusting it"
    )


def test_the_prompts_leg_is_a_registered_private_sqlite_owner():
    """The seam owner exists, is read-only-URI, and preserves source mode."""
    from tldw_chatbook.DB.private_sqlite import (
        SQLITE_OWNER_REGISTRY,
        SQLiteTargetKind,
    )

    policy = SQLITE_OWNER_REGISTRY["rag.prompts_keyword_leg"]
    assert policy.production_module == "tldw_chatbook/RAG_Search/simplified/rag_service"
    assert policy.allowed_target_kinds == {SQLiteTargetKind.READ_ONLY_URI}
    assert policy.preserve_read_only_source_mode is True
