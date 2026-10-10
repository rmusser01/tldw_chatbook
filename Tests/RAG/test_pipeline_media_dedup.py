"""TASK-407: unscoped media FTS results must survive pipeline dedup.

``search_media_db``'s broad row carries no ``content`` key (full transcripts
are deliberately kept out of the search SELECT), so ``search_media_fts5``
used to build ``SearchResult(content="")`` for EVERY media hit.
``deduplicate_results`` keys on ``content[:200]``, which collapsed all
unscoped multi-media results to exactly one regardless of how many matched.
These tests pin the repair: the media leg populates each hit with a bounded
content prefix (title fallback for content-less rows), so distinct matches
keep distinct dedup keys while true content duplicates still collapse.

Real tmp_path file-backed ``MediaDatabase`` throughout, mirroring
``Tests/RAG/test_scope_pipeline_enforcement.py``'s fixture patterns.
"""

from typing import List

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.pipeline_types import SearchResult

pytestmark = pytest.mark.unit

# Minimal pipeline shape exercising the real media leg plus the shared
# dedup process step -- the exact composition whose interaction collapsed
# every unscoped media search to one result.
_MEDIA_DEDUP_PIPELINE = {
    "name": "media-dedup-regression",
    "steps": [
        {
            "type": "parallel",
            "functions": [{"function": "search_media_fts5", "config": {"top_k": 10}}],
        },
        {"type": "process", "function": "deduplicate_results"},
    ],
}


class _App:
    """App double exposing the media_db seam the pipeline legs read."""

    def __init__(self, media_db):
        self.media_db = media_db
        self.chachanotes_db = None


@pytest.fixture()
def media_db(tmp_path):
    db = MediaDatabase(tmp_path / "media.db", client_id="task407-test")
    yield db
    try:
        db.close_connection()
    except Exception:
        pass


def _seed_media(db: MediaDatabase, n: int = 4) -> List[int]:
    ids = []
    for i in range(n):
        media_id, _uuid, _msg = db.add_media_with_keywords(
            title=f"Doc {i}",
            media_type="document",
            content=f"zanzibarite crystal sample {i}",
            keywords=["test"],
        )
        ids.append(media_id)
    return ids


class TestUnscopedMediaSearchSurvivesDedup:
    def test_search_media_db_row_has_no_content_key(self, media_db):
        """Premise pin: the broad search row omits ``content`` by design."""
        _seed_media(media_db)

        rows, _total = media_db.search_media_db(
            search_query="zanzibarite",
            search_fields=["title", "content"],
            page=1,
            results_per_page=20,
            include_trash=False,
        )

        assert len(rows) >= 3
        assert all("content" not in row for row in rows)

    async def test_media_leg_populates_nonempty_content(self, media_db):
        """The leg must not emit content="" placeholders for media hits."""
        ids = _seed_media(media_db)
        app = _App(media_db)

        results = await pfs.search_media_fts5(app, "zanzibarite", limit=10)

        assert {r.id for r in results} == {str(i) for i in ids}
        assert all(r.content.strip() for r in results), (
            [r.content for r in results]
        )
        # Content prefixes are distinct per hit, so dedup keys differ.
        assert len({r.content[:200] for r in results}) == len(results)

    async def test_three_plus_distinct_matches_survive_the_pipeline(self, media_db):
        """AC1: unscoped media FTS search keeps all distinct matches after
        the pipeline's deduplicate_results step (was: collapsed to 1)."""
        ids = _seed_media(media_db, n=4)
        assert len(ids) == 4
        app = _App(media_db)

        result_dicts, _context = await pbs.execute_pipeline(
            dict(_MEDIA_DEDUP_PIPELINE), app, "zanzibarite", {"media": True}
        )

        assert len(result_dicts) == 4, (
            f"expected all 4 distinct media matches through the pipeline, "
            f"got {len(result_dicts)}: {[r['id'] for r in result_dicts]}"
        )
        assert {r["id"] for r in result_dicts} == {str(i) for i in ids}


class TestDedupStillCollapsesTrueDuplicates:
    def test_identical_content_same_source_collapses(self):
        """AC2: content-prefix dedup semantics are unchanged within a source
        -- identical first-200-chars content still collapses to one result
        (highest score wins)."""
        dup_a = SearchResult(
            source="note", id="1", title="A", content="same text" * 10, score=0.1
        )
        dup_b = SearchResult(
            source="note", id="2", title="B", content="same text" * 10, score=0.9
        )

        deduped = pfs.deduplicate_results([dup_a, dup_b])

        assert len(deduped) == 1
        assert deduped[0].id == "2"  # higher score wins the shared key

    def test_identical_content_across_sources_survives(self):
        """A media item and a note sharing text are distinct results, not
        duplicates -- content keys are scoped by source (TASK-407: a bare
        content key ate one of the pair once media hits carried content)."""
        media_hit = SearchResult(
            source="media", id="1", title="Doc", content="shared zanzibarite text"
        )
        note_hit = SearchResult(
            source="note", id="1", title="Note", content="shared zanzibarite text"
        )

        deduped = pfs.deduplicate_results([media_hit, note_hit])

        assert {(r.source, r.id) for r in deduped} == {("media", "1"), ("note", "1")}

    async def test_identical_media_content_survives_only_once(self, media_db):
        """AC2 at the media seam: two media items whose content prefixes are
        identical are true duplicates and collapse to one through the
        pipeline."""
        for title in ("Twin A", "Twin B"):
            media_db.add_media_with_keywords(
                title=title,
                media_type="document",
                content="identical zanzibarite description",
                keywords=["test"],
            )
        app = _App(media_db)

        result_dicts, _context = await pbs.execute_pipeline(
            dict(_MEDIA_DEDUP_PIPELINE), app, "zanzibarite", {"media": True}
        )

        assert len(result_dicts) == 1, result_dicts

    async def test_contentless_media_falls_back_to_title(self, media_db):
        """Rows whose content is empty/NULL (legacy rows; ingestion itself
        rejects NULL and hash-dedups empty strings) must fall back to a
        title-based snippet, and distinct titles must not re-collapse -- the
        filed bug's content='' shape."""
        ids = _seed_media(media_db, n=3)
        conn = media_db.get_connection()
        # Bump version alongside content: Media's sync trigger requires a
        # version increment on any content-touching UPDATE.
        conn.execute(
            f"UPDATE Media SET content = '', version = version + 1 "
            f"WHERE id IN ({','.join('?' * len(ids))})",
            tuple(ids),
        )
        conn.commit()
        app = _App(media_db)

        results = await pfs.search_media_fts5(app, "Doc", limit=10)

        assert len(results) == 3
        assert all(r.content for r in results), [r.content for r in results]
        assert sorted(r.content for r in results) == ["Doc 0", "Doc 1", "Doc 2"]
        assert len({r.content[:200] for r in results}) == 3
