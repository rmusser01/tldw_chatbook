"""B9: ``search_media_db`` drops its redundant DISTINCT (1:1 joins proven).

Join-cardinality audit of every JOIN ``search_media_db`` can emit (the Step-1
gate this module documents):

* The builder appends exactly ONE join, ever: ``JOIN media_fts fts ON
  fts.rowid = m.id`` (single ``joins.append`` site, guarded to at most one
  entry). ``media_fts`` is an FTS5 external-content table over ``Media``
  with ``content_rowid='id'`` (Client_Media_DB_v2 schema), so its rowid
  space IS ``Media.id`` and the join is 1:1 (the FTS-first ``CROSS JOIN``
  count spelling is the same relation, reordered).
* Every keyword predicate (``must_have_keywords``, ``must_not_have_keywords``,
  the TASK-31274 keyword search branch) is a correlated scalar subquery /
  ``EXISTS`` in WHERE -- no row multiplication.
* ``_HAS_ANALYSIS_SELECT`` is a correlated ``EXISTS`` projection, not a join.

Therefore every result row is already a distinct Media row and
``SELECT DISTINCT`` / ``COUNT(DISTINCT m.id)`` only forced a dedup pass that
runs before LIMIT can short-circuit (pre-change EXPLAIN on a 1k-match
fixture showed ``USE TEMP B-TREE FOR DISTINCT`` on the page query).

The golden equivalence cases (text search, keyword filter, combined,
empty) were captured against the pre-change implementation and are pinned
below; ids are additionally asserted unique on a many-matches fixture.
"""

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture
def media_db():
    db = MediaDatabase(db_path=":memory:", client_id="b9_golden")
    yield db
    db.close_connection()


def _seed_golden(db: MediaDatabase) -> None:
    """Deterministic small fixture: overlapping keywords, FTS-indexed text."""
    rows = [
        # (title, content, keywords)
        ("Dragon Lore Alpha", "dragon alpha body text", [" fantasy ", "dragons"]),
        ("Dragon Lore Beta", "dragon beta body text", ["dragons"]),
        ("Dragon Lore Gamma", "gamma body without the word", [" fantasy "]),
        ("Unrelated Delta", "completely different content", ["sci-fi"]),
        ("Epsilon Dragon Guide", "a guide about dragons and lore", []),
        ("Zeta Manual", "operator manual text", ["manuals", " fantasy "]),
    ]
    for title, content, keywords in rows:
        media_id = db.add_media_with_keywords(
            title=title,
            media_type="pdf",
            content=content,
            keywords=keywords,
        )
        assert media_id is not None


@pytest.fixture
def golden_db(media_db):
    _seed_golden(media_db)
    return media_db


class TestGoldenEquivalence:
    """Golden result sets captured on the pre-change (DISTINCT) code."""

    def test_text_search(self, golden_db):
        results, total = golden_db.search_media_db(search_query="dragon")
        # Gamma matches via its TITLE ("Dragon Lore Gamma"); ordering is
        # last_modified DESC (seed order reversed).
        assert total == 4
        assert [r["title"] for r in results] == [
            "Epsilon Dragon Guide",
            "Dragon Lore Gamma",
            "Dragon Lore Beta",
            "Dragon Lore Alpha",
        ]
        assert len({r["id"] for r in results}) == len(results)

    def test_keyword_filter(self, golden_db):
        results, total = golden_db.search_media_db(
            search_query=None, must_have_keywords=["fantasy"]
        )
        assert total == 3
        assert {r["title"] for r in results} == {
            "Dragon Lore Alpha",
            "Dragon Lore Gamma",
            "Zeta Manual",
        }

    def test_combined_text_and_keyword(self, golden_db):
        results, total = golden_db.search_media_db(
            search_query="dragon",
            must_have_keywords=["dragons"],
            media_types=["pdf"],
        )
        assert total == 2
        assert {r["title"] for r in results} == {
            "Dragon Lore Alpha",
            "Dragon Lore Beta",
        }

    def test_empty_result(self, golden_db):
        results, total = golden_db.search_media_db(search_query="xyzzyplugh")
        assert (results, total) == ([], 0)

    def test_keyword_search_field(self, golden_db):
        # TASK-31274 branch: keywords as a search field (EXISTS OR-leg).
        results, total = golden_db.search_media_db(
            search_query="dragons", search_fields=["keywords"]
        )
        assert total == 2
        assert {r["title"] for r in results} == {
            "Dragon Lore Alpha",
            "Dragon Lore Beta",
        }


class TestNoDistinctDedup:
    """The dedup pass is gone from the emitted SQL and the page short-circuits.

    Evidence baseline (pre-change, this fixture, 1000 matches): the traced
    page query began ``SELECT DISTINCT m.id, ...`` and the full search cost
    21,121 VM steps (set_progress_handler, granularity 1) -- the DISTINCT
    dedup walked every match before LIMIT could short-circuit. Post-change
    the page SQL carries no DISTINCT and the step count drops materially.
    """

    @staticmethod
    def _seed_many(db: MediaDatabase, count: int) -> None:
        for i in range(count):
            db.add_media_with_keywords(
                title=f"Dragon volume {i}",
                media_type="pdf",
                content=f"dragon lore body {i}",
                keywords=[f"kw{i % 7}"],
            )

    def _captured_sql(self, db: MediaDatabase, **kwargs):
        statements = []

        def trace(statement: str) -> None:
            statements.append(statement)

        conn = db.get_connection()
        conn.set_trace_callback(trace)
        try:
            results, total = db.search_media_db(**kwargs)
        finally:
            conn.set_trace_callback(None)
        return results, total, statements

    def _page_sql(self, statements: list[str], *, fts: bool) -> str:
        page_sqls = [
            s
            for s in statements
            if s.lstrip().upper().startswith("SELECT")
            and "LIMIT" in s.upper()
            and ("media_fts" in s or not fts)
        ]
        assert page_sqls, "page query not captured"
        return page_sqls[-1]

    def test_fts_page_query_carries_no_distinct(self, media_db):
        self._seed_many(media_db, 1000)
        results, total, statements = self._captured_sql(
            media_db, search_query="dragon"
        )
        assert total == 1000
        assert len(results) == 20
        assert len({r["id"] for r in results}) == len(results)
        page_sql = self._page_sql(statements, fts=True)
        assert "DISTINCT" not in page_sql.upper(), (
            f"page query still dedups: {page_sql[:160]}..."
        )

    def test_browse_page_query_carries_no_distinct(self, media_db):
        self._seed_many(media_db, 1000)
        results, total, statements = self._captured_sql(
            media_db, search_query=None
        )
        assert total == 1000
        assert len(results) == 20
        page_sql = self._page_sql(statements, fts=False)
        assert "DISTINCT" not in page_sql.upper(), (
            f"page query still dedups: {page_sql[:160]}..."
        )

    def test_page_fetch_vm_steps_stay_bounded(self, media_db):
        """LIMIT must short-circuit the page leg.

        Evidence on this exact fixture (1000 matches, set_progress_handler
        granularity 1): 21,121 VM steps pre-change (DISTINCT dedup pass on
        both count and page legs) vs 17,950 post-change. The count leg keeps
        its full MATCH scan by contract (``total_matches`` is part of the
        API), so the delta IS the removed dedup pass. The bound sits between
        the two measured points.
        """
        self._seed_many(media_db, 1000)
        conn = media_db.get_connection()
        steps = {"n": 0}
        conn.set_progress_handler(lambda: (steps.__setitem__("n", steps["n"] + 1), 0)[1], 1)
        try:
            results, total = media_db.search_media_db(search_query="dragon")
        finally:
            conn.set_progress_handler(None, 0)
        assert total == 1000
        assert len(results) == 20
        assert steps["n"] < 19500, (
            f"{steps['n']} VM steps for a 20-row page over 1000 matches "
            "(pre-change DISTINCT baseline: 21,121)"
        )
