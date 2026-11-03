"""Installed heads extend every qualified legacy schema without relabeling it."""

import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA, recovery_adapters


@pytest.mark.parametrize(
    "leaf,owner_id,old,new",
    [
        ("chat.db", "db.chachanotes.primary", 81, 87),
        ("workspaces.db", "db.workspaces", 8, 11),
        ("runs.db", "db.agent_runs", 25, 27),
    ],
)
def test_real_legacy_source_heads_capture_and_migrate_exactly(
    tmp_path, leaf, owner_id, old, new
):
    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    path = source / leaf
    owner = next(
        item
        for item in (*core_adapters(), *recovery_adapters())
        if item.owner_id == owner_id
    )
    query = (
        "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        if leaf == "chat.db"
        else "SELECT MAX(version) FROM schema_version"
    )
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(query).fetchone()[0] == old
    before = path.read_bytes()
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    if leaf == "workspaces.db":
        # Restricted restore deliberately retains its native 8->9 refusal.
        assert validate_candidate(owner, path, Event(), migrate=True) == (
            "unsupported_schema_migration",
        )
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

        store = WorkspaceDB(path)
        store.close()
    else:
        assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(query).fetchone()[0] == new
        assert (
            conn.execute(
                "SELECT count(*) FROM mobile_consolidation_batches"
            ).fetchone()[0]
            == 0
        )
    assert validate_candidate(owner, path, Event(), migrate=False) == ()


@pytest.mark.parametrize(
    "catalog", [catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 25]
)
def test_every_qualified_v25_layout_migrates_to_exact_v27(tmp_path, catalog):
    path = tmp_path / "agent_runs.db"
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.execute("INSERT INTO schema_version VALUES (25)")
        conn.commit()
    path.chmod(0o600)
    owner = next(
        item for item in recovery_adapters() if item.owner_id == "db.agent_runs"
    )
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    assert validate_candidate(owner, path, Event(), migrate=False) == ()


def test_owner_migration_failure_rolls_back_new_tables_and_version(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB import mobile_import_authority
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    path = source / "runs.db"
    monkeypatch.setattr(
        mobile_import_authority,
        "AGENT_RUNS_MIGRATION",
        (
            *mobile_import_authority.AGENT_RUNS_MIGRATION,
            "SELECT * FROM deliberately_missing_table",
        ),
    )
    with (
        ConsoleExecutionOwner.acquire(path) as owner,
        pytest.raises(sqlite3.OperationalError, match="deliberately_missing"),
    ):
        AgentRunsDB(path, execution_owner=owner)
    with closing(sqlite3.connect(path)) as conn:
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 26
        )
        assert (
            conn.execute(
                "SELECT name FROM sqlite_schema WHERE name LIKE 'mobile_%'"
            ).fetchall()
            == []
        )


def test_all_registered_native_mobile_column_projections(tmp_path):
    """Inventory every installed physical variant and its explicit disposition."""
    import hashlib
    import json

    from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import (
        _FAMILIES,
        _native_column_layouts,
    )
    from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES
    from tldw_chatbook.DB import recovery_core_schema, recovery_operations
    from tldw_chatbook.DB.ChaChaNotes_DB import (
        _install_canvas_revision_payload_validator,
    )

    assert next(row[3:] for row in _FAMILIES if row[0] == "chat") == (
        (81, 82, 83, 84, 85),
        87,
    )
    specs = {(s.family, s.table): s for s in _TABLES}
    schemas = [
        ("runs", v, c, None) for v, c in _AGENT_RUNS_SCHEMA if v in (24, 25, 26, 27)
    ]
    schemas += [
        ("workspaces", v, c, None)
        for v, c in recovery_operations._WORKSPACES_SCHEMA
        if v in (8, 9, 10, 11)
    ]
    for version in range(81, 88):
        schemas += [
            ("chat", version, c, None)
            for c in getattr(recovery_core_schema, f"CHACHANOTES_V{version}_SCHEMAS")
        ]
    for version, name in (
        (81, "CANVAS_V81"),
        (82, "CANVAS_V82"),
        (83, "PORTRAIT_V83"),
        (84, "REGIONS_V84"),
        (85, "ARCHIVE_V85"),
        (86, "MIXINS_V86"),
        (87, "MOBILE_V87"),
    ):
        for catalog in getattr(recovery_operations, f"_SUBSCRIPTIONS_{name}_SCHEMAS"):
            for subscriptions_version in (2, 3):
                assert (
                    subscriptions_version,
                    catalog,
                ) in recovery_operations._SUBSCRIPTIONS_SCHEMA
                schemas.append(
                    ("shared_subscriptions", version, catalog, subscriptions_version)
                )
    report, failures = [], []
    for family, version, catalog, subscriptions_version in schemas:
        shared = family == "shared_subscriptions"
        projected_family = "chat" if shared else family
        target = {"chat": 87, "workspaces": 11, "runs": 27}[projected_family]
        purpose = "canonical" if version == target else "legacy"
        with closing(sqlite3.connect(":memory:")) as conn:
            _install_canvas_revision_payload_validator(conn)
            for sql in catalog:
                if sql.startswith("CREATE VIRTUAL TABLE"):
                    conn.execute(sql)
            created = {r[0] for r in conn.execute("SELECT name FROM sqlite_schema")}
            for sql in catalog:
                if not sql.startswith("CREATE TABLE"):
                    continue
                name = sql.split("(", 1)[0].split()[-1].strip("\"'`[]")
                if name not in created and name != "sqlite_sequence":
                    conn.execute(sql)
            tables = {}
            for (table,) in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            ).fetchall():
                metadata = tuple(
                    r
                    for r in conn.execute('PRAGMA table_xinfo("' + table + '")')
                    if r[6] != 1
                )
                columns = tuple(r[1] for r in metadata)
                spec = specs.get((projected_family, table))
                assert spec is not None or shared, (family, version, table)
                if (not shared or spec is not None) and (
                    spec is None
                    or columns
                    not in _native_column_layouts(
                        purpose, projected_family, table, spec.columns, version
                    )
                ):
                    failures.append(
                        (
                            family,
                            version,
                            table,
                            columns,
                            None if spec is None else spec.columns,
                        )
                    )
                defaults = {}
                if spec is not None:
                    for column in set(spec.columns) - set(columns):
                        assert column in {
                            "archived_at",
                            "archive_revision",
                            "review_boundary_id",
                            "composition_recipe_json",
                            "mixin_snapshot_schema_version",
                        }
                        defaults[column] = 0 if column == "archive_revision" else None
                tables[table] = {
                    "ordered_columns": columns,
                    "sqlite_declarations": [
                        dict(
                            zip(
                                (
                                    "cid",
                                    "name",
                                    "type",
                                    "not_null",
                                    "default_sql",
                                    "pk",
                                    "hidden",
                                ),
                                row,
                            )
                        )
                        for row in metadata
                    ],
                    "disposition": "native_shared_backup_preserved_no_mobile_source_grant"
                    if shared or (family, version) in {("runs", 26), ("chat", 86)}
                    else (
                        "canonical_preserved"
                        if purpose == "canonical"
                        else spec.handler
                    ),
                    "missing_column_projection": defaults,
                }
            missing = {}
            if not shared:
                for (spec_family, table), spec in specs.items():
                    if (
                        spec_family == family
                        and table not in tables
                        and not spec.canonical_only
                    ):
                        missing[table] = "authenticated_predecessor_absence"
            report.append(
                {
                    "family": family,
                    "version": version,
                    "subscriptions_version": subscriptions_version,
                    "mobile_member": "none"
                    if shared or (family, version) in {("runs", 26), ("chat", 86)}
                    else purpose,
                    "catalog_sha256": hashlib.sha256(
                        repr(catalog).encode()
                    ).hexdigest(),
                    "tables": tables,
                    "absent_tables": missing,
                }
            )
    destination = tmp_path / "registered-native-column-inventory.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    assert not failures, failures


@pytest.mark.parametrize("route", ["runtime", "recovery"])
def test_native_mixin_v86_migrates_to_mobile_v87_without_losing_attempts(
    tmp_path, route
):
    from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
    from Tests.DB.test_console_prompt_mixins import attempt, saved_turn
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.console_prompt_mixins import read_attempt, write_attempt
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

    source = tmp_path / "native-v86.sqlite"
    with chachanotes_db_at_version(source, 86) as db:
        conversation_id, messages = saved_turn(db)
        with db.transaction() as cursor:
            write_attempt(
                cursor, conversation_id, batch=attempt(), native_message_ids=messages
            )
    original = source.read_bytes()
    candidate = tmp_path / "mobile-v87.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", source, candidate)
    if route == "runtime":
        db = CharactersRAGDB(candidate, "mobile-mixin-migration")
        db.close()
    else:
        owner = next(
            owner
            for owner in core_adapters()
            if owner.owner_id == "db.chachanotes.primary"
        )
        assert validate_candidate(owner, candidate, Event(), migrate=True) == ()
    with closing(sqlite3.connect(candidate)) as connection:
        assert (
            connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 87
        )
        assert (
            read_attempt(
                connection.cursor(), conversation_id, attempt_id="attempt"
            ).snapshot
            == attempt().snapshot
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM mobile_consolidation_batches"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert source.read_bytes() == original
