"""Regenerate ADR-285's exact workflow catalog data from installed predecessors.

Run with the repository on PYTHONPATH and redirect stdout to a temporary file.
Compare the result before replacing the checked-in declaration.
"""

from tldw_chatbook.Backup_Recovery.sqlite_validation import _reference
from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA
from tldw_chatbook.DB.recovery_review_boundary_schema import review_boundary_catalogs
from tldw_chatbook.DB.recovery_workflow_legacy_schema import LEGACY_WORKFLOW_SCHEMAS
from tldw_chatbook.DB.recovery_workflow_schema import (
    workflow_admission_catalog,
    workflow_catalog,
    workflow_checkpoint_catalog,
    workflow_repair_input_catalog,
    workflow_window_catalog,
)


def originals():
    """Follow the original installed derivation, including its ordering."""
    reviews = tuple(
        dict.fromkeys(
            (25, variant)
            for version, schema in _AGENT_RUNS_SCHEMA
            if version == 24
            for variant in review_boundary_catalogs(schema)
        )
    )
    canonical = tuple((26, workflow_catalog(schema)) for _, schema in reviews)
    canonical += tuple(
        (version + 1, review_boundary_catalogs(schema)[0])
        for version, schema in LEGACY_WORKFLOW_SCHEMAS
    )
    for start, step in enumerate(
        (
            workflow_admission_catalog,
            workflow_checkpoint_catalog,
            workflow_window_catalog,
            workflow_repair_input_catalog,
        ),
        26,
    ):
        canonical += tuple(
            (start + 1, step(schema)) for v, schema in canonical if v == start
        )
    return reviews, tuple(dict.fromkeys(canonical)), LEGACY_WORKFLOW_SCHEMAS


def historical_schemas():
    """Upgrade authentic empty historical producers; deduplicate only after all five routes."""
    import sqlite3
    from contextlib import closing, nullcontext

    from Tests.DB.fixtures.agent_runs_history import install_historical
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _catalog, _metadata
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    outcomes = {}
    for version in (12, 15, 17, 18, 21):
        with closing(
            connect_private_sqlite("recovery.workflow_catalog", ":memory:")
        ) as conn:
            conn.isolation_level = None
            install_historical(conn, version)
            # Run the installed constructor DDL on the private memory file.
            db = AgentRunsDB.__new__(AgentRunsDB)
            db.connection = lambda: nullcontext(conn)
            conn.row_factory = sqlite3.Row
            db._initialize_schema()
            conn.row_factory = None
            rows = _catalog(conn)
            schema = tuple(row[3] for row in rows if row[3] is not None)
            assert _reference(schema) == (rows, _metadata(conn, rows))
            assert [
                row[0]
                for row in conn.execute(
                    "SELECT version FROM schema_version ORDER BY version"
                )
            ] == list(range(4, 31))
            outcomes[30, schema] = None
    assert len(outcomes) == 3
    return tuple(outcomes)


def render():
    """Deduplicate physical objects with their installed metadata."""
    pool = {}
    profiles = []
    for family in (*originals(), historical_schemas()):
        group = []
        for version, schema in family:
            rows, metadata = _reference(schema)
            objects = {entry[0]: entry for entry in metadata}
            indices = []
            for row in rows:
                key = (row, objects.get(row[1]))
                indices.append(pool.setdefault(key, len(pool)))
            group.append((version, tuple(indices)))
        profiles.append(group)
    lines = [
        '"""Installed ADR-285 data. Regenerate with scripts/freeze_workflow_catalogs.py."""',
        "",
        "_OBJECTS = (",
        *(f"    {obj!r}," for obj in pool),
        ")",
        "",
    ]
    for name, group in zip(
        ("REVIEW", "CANONICAL", "LEGACY", "HISTORICAL"), profiles, strict=True
    ):
        lines += [f"{name} = (", *(f"    {item!r}," for item in group), ")", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(), end="")
