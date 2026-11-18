"""Exact AgentRuns v24-to-v25 review-boundary catalog delta (ADR-281)."""

import re

REVIEW_BOUNDARY_TABLE = (
    "CREATE TABLE automatic_review_boundary_claims (\n"
    "    boundary_id TEXT PRIMARY KEY,\n"
    "    attempt_id TEXT NOT NULL UNIQUE REFERENCES automatic_wake_attempts(id),\n"
    "    owner_id TEXT NOT NULL,\n"
    "    chain_id TEXT NOT NULL REFERENCES automatic_work_chains(id),\n"
    "    review_id TEXT NOT NULL,\n"
    "    slot_id TEXT NOT NULL,\n"
    "    slot_attempt_id TEXT NOT NULL,\n"
    "    source_epoch INTEGER NOT NULL,\n"
    "    context_revision INTEGER NOT NULL,\n"
    "    grant_id TEXT NOT NULL,\n"
    "    target_fingerprint TEXT NOT NULL,\n"
    "    team_fingerprint TEXT NOT NULL,\n"
    "    override_fingerprint TEXT NOT NULL,\n"
    "    predecessor_run_ids_json TEXT NOT NULL,\n"
    "    predecessor_settlement_ids_json TEXT NOT NULL,\n"
    "    UNIQUE(owner_id, chain_id, review_id, slot_id, slot_attempt_id)\n"
    ")"
)
REVIEW_BOUNDARY_COLUMN = "review_boundary_id TEXT"
REVIEW_BOUNDARY_AUTO_INDEXES = frozenset(
    (
        f"sqlite_autoindex_automatic_review_boundary_claims_{n}",
        "automatic_review_boundary_claims",
    )
    for n in (1, 2, 3)
)
_GOAL_COLUMN = ", attempt_kind TEXT NOT NULL DEFAULT 'fleet_wake'"


def _key(sql):
    match = re.match(
        r'CREATE (?:UNIQUE )?(TABLE|INDEX|TRIGGER|VIEW) ["`]?([^\s"`(]+)', sql
    )
    return match[1].lower(), match[2]


def review_boundary_catalogs(catalog):
    """Both physical v25 lineages of one v24 catalog.

    A v24 database upgraded to v25 appends the column after the goal column;
    the current constructor adds it before the goal column (fresh databases
    and pre-v24 databases upgraded in one step).
    """

    def with_wake(change):
        rows = tuple(
            change(sql)
            if sql.startswith("CREATE TABLE automatic_wake_attempts ")
            else sql
            for sql in catalog
        )
        return tuple(sorted((*rows, REVIEW_BOUNDARY_TABLE), key=_key))

    return (
        with_wake(lambda sql: sql[:-1] + ", " + REVIEW_BOUNDARY_COLUMN + ")"),
        with_wake(
            lambda sql: sql.replace(
                _GOAL_COLUMN, ", " + REVIEW_BOUNDARY_COLUMN + _GOAL_COLUMN, 1
            )
        ),
    )


REVIEW_BOUNDARY_MIGRATION = (
    "ALTER TABLE automatic_wake_attempts ADD COLUMN " + REVIEW_BOUNDARY_COLUMN,
    REVIEW_BOUNDARY_TABLE,
    "INSERT INTO schema_version(version) VALUES (25)",
)
