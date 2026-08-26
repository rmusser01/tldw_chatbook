"""The Samira boot preflight asks SQLite, and gets the same answer (TASK-21111(d)).

`ensure_builtin_samira` runs on every boot. Its card lookup used to read the
whole `character_cards` table into Python and `json.loads` every row's
`extensions`. The targeted `json_extract` query replaces it -- but only if it
agrees with the scan on the awkward rows the scan tolerated, which is what
these tests pin. `_find_builtin_samira_card_by_scan` is retained as the
JSON1-less fallback and doubles as the oracle here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from tldw_chatbook.Character_Chat import visual_identity

SAMIRA_EXTENSIONS = json.dumps({"tldw/builtin_id": "samira"})


class _Db:
    """Minimal `execute_query` seam, counting the queries issued."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.queries: list[str] = []

    def execute_query(self, sql, params=()):
        self.queries.append(sql)
        return self.conn.execute(sql, params)


def _db(rows: list[tuple[str, str | None]]) -> _Db:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE character_cards (id INTEGER PRIMARY KEY, name TEXT,"
        " extensions TEXT, deleted INTEGER DEFAULT 0)"
    )
    conn.executemany(
        "INSERT INTO character_cards (name, extensions) VALUES (?, ?)", rows
    )
    conn.commit()
    return _Db(conn)


AWKWARD: list[tuple[str, str | None]] = [
    ("null ext", None),
    ("empty ext", ""),
    ("malformed", "{not json"),
    ("array ext", "[1, 2, 3]"),
    ("scalar ext", '"just a string"'),
    ("nan constant", '{"tldw/builtin_id": NaN}'),
    ("numeric id", '{"tldw/builtin_id": 7}'),
    ("other builtin", '{"tldw/builtin_id": "someone-else"}'),
    ("nested only", '{"data": {"tldw/builtin_id": "samira"}}'),
]


@pytest.mark.parametrize("with_samira", [False, True])
def test_the_targeted_query_matches_the_scan_on_awkward_rows(
    with_samira: bool,
) -> None:
    rows = list(AWKWARD)
    if with_samira:
        rows.append(("Samira", SAMIRA_EXTENSIONS))
    db = _db(rows)

    targeted = visual_identity._find_builtin_samira_card(db)
    scanned = visual_identity._find_builtin_samira_card_by_scan(db)

    assert (targeted is None) == (scanned is None)
    if scanned is not None:
        assert targeted == scanned
        assert targeted["name"] == "Samira"
    else:
        assert targeted is None


def test_a_malformed_extensions_row_does_not_abort_the_lookup() -> None:
    """`json_extract` raises on malformed JSON; the guard must run first.

    A single corrupt `extensions` value would otherwise take out the whole
    boot preflight, where the Python scan simply skipped that row.

    The single-query assertion is the part that matters: without it, a
    guard-less query would still "pass" by raising and falling back to the
    scan -- i.e. by reintroducing the whole-table read this change removes,
    silently, on every boot of the affected profile. (Both a guard-less and
    an `AND`-guarded mutant were tried; only this assertion tells them apart.)
    """
    db = _db([("broken", "{not json"), ("Samira", SAMIRA_EXTENSIONS)])

    found = visual_identity._find_builtin_samira_card(db)

    assert found is not None and found["name"] == "Samira"
    assert len(db.queries) == 1, "the malformed row forced a fallback full scan"


def test_the_lowest_id_wins_when_several_rows_claim_the_builtin_id() -> None:
    db = _db(
        [
            ("first", SAMIRA_EXTENSIONS),
            ("second", SAMIRA_EXTENSIONS),
        ]
    )

    assert visual_identity._find_builtin_samira_card(db)["name"] == "first"


def test_the_lookup_issues_exactly_one_query() -> None:
    """No per-row work: one statement, and SQLite stops at the first hit."""
    db = _db([("Samira", SAMIRA_EXTENSIONS)])

    visual_identity._find_builtin_samira_card(db)

    assert len(db.queries) == 1
    assert "LIMIT 1" in db.queries[0]


def test_a_json1_less_sqlite_falls_back_to_the_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback keeps the boot seed working on an exotic SQLite build."""
    db = _db([("Samira", SAMIRA_EXTENSIONS)])
    real_execute = db.execute_query
    calls: list[str] = []

    def failing_json1(sql, params=()):
        calls.append(sql)
        if "json_extract" in sql:
            raise sqlite3.OperationalError("no such function: json_extract")
        return real_execute(sql, params)

    monkeypatch.setattr(db, "execute_query", failing_json1)

    found = visual_identity._find_builtin_samira_card(db)

    assert found is not None and found["name"] == "Samira"
    assert any("json_extract" in sql for sql in calls)
    assert any("json_extract" not in sql for sql in calls)
