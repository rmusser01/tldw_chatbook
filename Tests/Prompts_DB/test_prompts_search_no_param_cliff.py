"""search_prompts must not bind one SQL variable per matching id (TASK-32804.10).

It materialised every matching prompt id in Python and bound them as one
`p.id IN (?,?,…)` list, so cost grew linearly with the match count and the call
hard-failed with "too many SQL variables" past SQLITE_LIMIT_VARIABLE_NUMBER
(32766). The id set is now kept in SQLite via a subquery, exactly as
search_library_prompts_page does.
"""

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase


@pytest.fixture
def db():
    d = PromptsDatabase(":memory:", client_id="cliff-test")
    yield d
    d.close_connection()


def _seed(db, n, term="dragonfruit"):
    for i in range(n):
        db.add_prompt(f"Prompt {i}", "Author", f"details about {term} number {i}")


def test_search_returns_all_matches_and_paginates(db):
    _seed(db, 40)
    page, total = db.search_prompts("dragonfruit", page=1, results_per_page=20)
    assert total == 40
    assert len(page) == 20


def test_search_query_binds_no_per_id_placeholder_list(db):
    """The cliff mechanism: prove the search query uses a subquery, not a
    `p.id IN (?,?,…)` placeholder list sized to the match count."""
    _seed(db, 40)

    executed = []
    real = db.execute_query

    def spy(query, params=None, *a, **k):
        executed.append((query, params))
        return real(query, params, *a, **k) if params is not None else real(query, *a, **k)

    db.execute_query = spy  # type: ignore[method-assign]
    try:
        page, total = db.search_prompts("dragonfruit", page=1, results_per_page=20)
    finally:
        db.execute_query = real  # type: ignore[method-assign]

    assert total == 40 and len(page) == 20
    # No executed statement may carry a bound-placeholder id list; the fix uses
    # `p.id IN (SELECT ...)`, so `IN (?` must not appear in the p.id condition.
    for query, params in executed:
        if "FROM Prompts p" in query and "p.id IN (" in query:
            assert "p.id IN (SELECT" in query, query
            assert "p.id IN (?" not in query, query
    # And no single statement binds ~40 params (the old IN list would).
    assert max((len(p) for _q, p in executed if p), default=0) < 10
