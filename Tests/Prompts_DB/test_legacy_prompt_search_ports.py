"""B8: legacy Prompts_DB searches keep golden results while dropping N+1.

Golden baseline captured on the PRE-PORT implementation (fetch-all FTS
rowids + ``IN (…)`` + per-row ``fetch_keywords_for_prompt`` loop): the ported
``search_prompts_by_text`` / ``search_prompts_by_keyword`` /
``search_prompts`` keyword attach must return byte-identical
(id/name/keyword) tuples for content, keyword, combined, and empty-result
searches over the same deterministic fixture.

New behavioral guarantees added by the port (kept separate from the golden
assertions): default page size 20 (matching ``search_prompts``'s
``results_per_page`` default), explicit ``limit``/``offset`` parameters, and
exactly ONE keywords query per page (no per-row N+1).
"""

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

TOTAL_PROMPTS = 32

#: 20 prompts whose details mention "review"; deterministic keywords each.
REVIEW_PROMPTS = [
    (f"p{i:02d} review", [f"kw-{i:02d}a", f"kw-{i:02d}b"]) for i in range(20)
]
#: 6 prompts carrying the "special" keyword; two also mention "review".
SPECIAL_PROMPTS = [
    ("k00 special", ["special"]),
    ("k01 special review", ["special"]),
    ("k02 special", ["special", "zz-extra"]),
    ("k03 special review", ["special"]),
    ("k04 special", ["special"]),
    ("k05 special", ["special"]),
]
#: 6 unrelated prompts.
UNRELATED_PROMPTS = [(f"q{i:02d} filler", [f"filler-{i:02d}"]) for i in range(6)]


def _seed(db: PromptsDatabase) -> None:
    rows = REVIEW_PROMPTS + SPECIAL_PROMPTS + UNRELATED_PROMPTS
    assert len(rows) == TOTAL_PROMPTS
    for i, (name, keywords) in enumerate(rows):
        details = (
            "needs careful review of the data"
            if "review" in name
            else "no match text here"
        )
        prompt_id, _, _ = db.add_prompt(
            name=name,
            author="tester",
            details=details,
            system_prompt="sys",
            user_prompt="usr",
            keywords=keywords,
        )
        assert prompt_id is not None
        # Deterministic recency: later seeds are "newer". The version bump
        # satisfies the Prompts sync-update trigger (it rejects in-place
        # last_modified edits without a version increment).
        stamp = f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z"
        with db.transaction() as conn:
            conn.execute(
                "UPDATE Prompts SET last_modified = ?, version = version + 1 WHERE id = ?",
                (stamp, prompt_id),
            )


@pytest.fixture
def seeded_db():
    db = PromptsDatabase(":memory:", client_id="b8_golden")
    _seed(db)
    yield db
    db.close_connection()


def _tuples(rows):
    return [(r["name"], tuple(r["keywords"])) for r in rows]


#: Every prompt whose FTS-indexed text contains "review": the 20 review
#: prompts plus the two "special" prompts with "review" in their NAME
#: (prompts_fts indexes name too, and search_prompts' text leg is an
#: unscoped MATCH -- golden-captured on the pre-port implementation).
REVIEW_TEXT_MATCHES = sorted(
    REVIEW_PROMPTS + [p for p in SPECIAL_PROMPTS if "review" in p[0]],
    key=lambda t: t[0].casefold(),
)

#: search_prompts orders by last_modified DESC (seed order reversed); its
#: default page (20) is the first 20 of the 22 recency-ordered matches.
_COMBINED_PAGE_NAMES = {
    n
    for n in list(
        reversed(
            [n for n, _ in REVIEW_PROMPTS]
            + [n for n, _ in SPECIAL_PROMPTS if "review" in n]
        )
    )[:20]
}


class TestGoldenEquivalence:
    """Golden results on the fixture; must hold pre- and post-port."""

    def test_content_match(self, seeded_db):
        rows = seeded_db.search_prompts_by_text("review", limit=1000)
        # Golden: pre-port code returned ALL matches, name COLLATE NOCASE
        # (the port's default page size caps the DEFAULT call at 20; the
        # full-set equivalence is pinned here with an explicit limit).
        assert _tuples(rows) == _tuples(
            [{"name": n, "keywords": k} for n, k in REVIEW_TEXT_MATCHES]
        )
        assert len(rows) == 22

    def test_keyword_match(self, seeded_db):
        rows = seeded_db.search_prompts_by_keyword("special")
        got = _tuples(rows)
        expected = sorted(_tuples(
            [{"name": n, "keywords": k} for n, k in SPECIAL_PROMPTS]
        ))
        assert got == expected
        assert len(got) == 6

    def test_combined_text_and_keywords(self, seeded_db):
        rows, total = seeded_db.search_prompts(
            "review", search_fields=["details", "keywords"]
        )
        names = {r["name"] for r in rows}
        # Golden: the text leg is an unscoped prompts_fts MATCH (field
        # selection does not scope the FTS columns), so the two
        # name-contains-review specials match too; the keyword leg ("review"
        # against prompt keywords) adds nothing on this fixture.
        assert total == 22
        # Default page size 20 over last_modified DESC order.
        assert names == _COMBINED_PAGE_NAMES
        by_name = dict(REVIEW_TEXT_MATCHES)
        for r in rows:
            assert tuple(r["keywords"]) == tuple(by_name[r["name"]])

    def test_combined_via_keyword_field(self, seeded_db):
        rows, total = seeded_db.search_prompts(
            "special", search_fields=["keywords"]
        )
        assert total == 6
        assert {r["name"] for r in rows} == {n for n, _ in SPECIAL_PROMPTS}

    def test_empty_result(self, seeded_db):
        assert seeded_db.search_prompts_by_text("xyzzyplugh") == []
        assert seeded_db.search_prompts_by_keyword("nosuchkeyword") == []
        rows, total = seeded_db.search_prompts("xyzzyplugh")
        assert (rows, total) == ([], 0)

    def test_search_prompts_keyword_attach_matches_seeded(self, seeded_db):
        rows, _ = seeded_db.search_prompts(
            "review", search_fields=["details"], results_per_page=5, page=1
        )
        by_name = dict(REVIEW_TEXT_MATCHES)
        for r in rows:
            assert tuple(r["keywords"]) == tuple(by_name[r["name"]])


class TestPortedBehavior:
    """Behavior ADDED by the port (limit/offset, bounded statement count)."""

    def test_default_page_size_is_twenty(self, seeded_db):
        # 22 text matches on the fixture; the ported default page size (20,
        # search_prompts' results_per_page default) caps the return.
        rows = seeded_db.search_prompts_by_text("review")
        assert len(rows) == 20
        names = [r["name"] for r in rows]
        assert names == sorted(names, key=str.casefold)
        assert names == [n for n, _ in REVIEW_TEXT_MATCHES][:20]

        rows2 = seeded_db.search_prompts_by_keyword("special")
        assert len(rows2) == 6  # under the default page size

    def test_limit_and_offset_honored(self, seeded_db):
        page1 = seeded_db.search_prompts_by_text("review", limit=5, offset=0)
        page2 = seeded_db.search_prompts_by_text("review", limit=5, offset=5)
        assert len(page1) == 5 and len(page2) == 5
        assert {r["name"] for r in page1}.isdisjoint({r["name"] for r in page2})
        # ordering by name NOCASE is stable across pages
        names = [r["name"] for r in page1] + [r["name"] for r in page2]
        assert names == sorted(names, key=str.casefold)

        kw_page = seeded_db.search_prompts_by_keyword("special", limit=4, offset=2)
        assert len(kw_page) == 4

    def test_by_text_page_of_twenty_uses_single_keywords_query(self, seeded_db):
        calls = []
        real_helper = PromptsDatabase._library_keywords_for_prompts

        def counting_helper(self, conn, prompt_ids):
            calls.append(list(prompt_ids))
            return real_helper(self, conn, prompt_ids)

        statements = []

        def trace(statement: str) -> None:
            statements.append(statement)

        conn = seeded_db.get_connection()
        conn.set_trace_callback(trace)
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    PromptsDatabase,
                    "_library_keywords_for_prompts",
                    counting_helper,
                )
                rows = seeded_db.search_prompts_by_text("review")
        finally:
            conn.set_trace_callback(None)
        assert len(rows) == 20
        assert len(calls) == 1, (
            f"expected ONE batched keywords query, got {len(calls)} attach rounds"
        )
        assert len(calls[0]) == 20
        # Bounded statement count overall: page + keywords (+ any pragma),
        # never 1 + page_size.
        select_count = sum(
            1 for s in statements if s.lstrip().upper().startswith("SELECT")
        )
        assert select_count <= 3, (
            f"page search issued {select_count} SELECTs; expected <= 3"
        )

    def test_by_keyword_page_uses_single_keywords_query(self, seeded_db):
        calls = []
        real_helper = PromptsDatabase._library_keywords_for_prompts

        def counting_helper(self, conn, prompt_ids):
            calls.append(list(prompt_ids))
            return real_helper(self, conn, prompt_ids)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                PromptsDatabase, "_library_keywords_for_prompts", counting_helper
            )
            rows = seeded_db.search_prompts_by_keyword("special")
        assert rows
        assert len(calls) == 1, (
            f"expected ONE batched keywords query, got {len(calls)} attach rounds"
        )
        assert len(calls[0]) == len(rows)


@pytest.mark.parametrize(
    "method", ["search_prompts", "search_prompts_by_text", "search_prompts_by_keyword"]
)
def test_keyword_batch_respects_the_live_variable_limit(seeded_db, method):
    import sqlite3

    seeded_db.get_connection().setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 3)
    if method == "search_prompts":
        rows, total = seeded_db.search_prompts("review")
        assert total == 22
        assert len(rows) == 20
    elif method == "search_prompts_by_text":
        rows = seeded_db.search_prompts_by_text("review")
        assert len(rows) == 20
    else:
        rows = seeded_db.search_prompts_by_keyword("special", limit=20)
        assert len(rows) == 6
    for row in rows:
        assert row["keywords"] == seeded_db.fetch_keywords_for_prompt(row["id"])
