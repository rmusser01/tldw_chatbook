"""Imported activations are inert; rollback keeps the original SQLite state."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Chat.response_rules_fixtures import inputs
from Tests.Chat.response_rules_store_fixtures import activate
from Tests.Chat.response_rules_store_fixtures import (
    rule_store as rule_store,  # noqa: PLC0414 - pytest fixture registration
)
from tldw_chatbook.Chat.response_rules.evaluator import aggregate_checks
from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_rules


def test_restore_imports_inactive_without_pending_work(rule_store):
    store, db, origin = rule_store
    binding = activate(store, origin)
    store.save_assessment(
        replace(aggregate_checks(origin, (), inputs=inputs()), state="pending")
    )
    with db.transaction() as cursor:
        prepare_imported_rules(cursor)
    assert store.list_bindings(binding.scope)[0].state == "disabled"
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_response_rule_assessments WHERE state='pending'"
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_machine_followup_receipts"
            ).fetchone()[0]
            == 0
        )


def test_internal_rollback_preserves_prior_binding_state(rule_store):
    store, db, origin = rule_store
    binding = activate(store, origin)
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        prepare_imported_rules(cursor)
        raise RuntimeError("rollback")
    assert store.list_bindings(binding.scope) == (binding,)


def test_installed_recovery_policy_migrates_only_disposable_candidate(
    rule_store, tmp_path
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters

    _, db, _ = rule_store
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    assert db._CURRENT_SCHEMA_VERSION in owner.schema_policy().versions
    assert validate_candidate(owner, db.db_path, Event(), migrate=False) == ()


# This child intentionally imports no test fixtures or runtime constructors.
# The producer runs in pytest's process; recovery starts with a fresh interpreter.
_COLD_IMPORT = r"""
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Event

from tldw_chatbook.DB.recovery_response_rules import prepare_imported_candidate
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.Backup_Recovery.sqlite_validation import _active_restrictions

def assert_cold():
    assert 'tldw_chatbook.config' not in sys.modules, 'recovery imported startup config'
    assert 'tldw_chatbook.Chat' not in sys.modules, 'recovery imported Chat runtime'

assert_cold()
candidate, outcome = Path(sys.argv[1]), sys.argv[2]
cancel = Event()
if outcome == 'cancelled':
    cancel.set()
opened = []
original = private_sqlite.open_recovery_validation
@contextmanager
def observe(*args, **kwargs):
    with original(*args, **kwargs) as value:
        connection, restrictions = value
        opened.append((connection, restrictions))
        if outcome == 'interrupted' and kwargs.get('writable'):
            def trace(sql):
                if sql.startswith('SELECT assessment_id,assessment_json'):
                    assert connection.total_changes > 0
                    cancel.set()
                    connection.set_progress_handler(restrictions.progress, 1)
            connection.set_trace_callback(trace)
        yield value
private_sqlite.open_recovery_validation = observe
try:
    prepare_imported_candidate(candidate, cancel)
except (TypeError, ValueError, sqlite3.DatabaseError) as error:
    assert outcome != 'valid', repr(error)
    if outcome == 'malformed':
        assert isinstance(error, TypeError), repr(error)
    elif outcome == 'invalid_json':
        # SQLite versions may detect this in quick_check or the JSON transform.
        assert isinstance(error, json.JSONDecodeError) or (
            isinstance(error, ValueError) and str(error) == 'invalid_sqlite_integrity'
        ), repr(error)
    elif outcome == 'cancelled':
        assert isinstance(error, ValueError) and str(error) == 'cancelled', repr(error)
    else:
        assert cancel.is_set() and isinstance(error, sqlite3.DatabaseError), repr(error)
else:
    assert outcome == 'valid', outcome
assert_cold()
assert _active_restrictions.get() is None
assert len(opened) in ({0} if outcome == 'cancelled' else {1, 2} if outcome == 'invalid_json' else {2})
for connection, restrictions in opened:
    assert not restrictions.preparing_response_rules
    try:
        connection.execute('SELECT 1')
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError('recovery connection remained physically open')
print(json.dumps({'outcome': outcome, 'closed_connections': len(opened), 'constructor_free': True}))
"""


@pytest.mark.parametrize(
    "outcome", ["valid", "malformed", "invalid_json", "cancelled", "interrupted"]
)
def test_restricted_import_preparation_disables_only_qualified_selected_payload(
    rule_store, tmp_path, outcome
):
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

    store, db, origin = rule_store
    binding = activate(store, origin)
    assessment = replace(aggregate_checks(origin, (), inputs=inputs()), state="pending")
    store.save_assessment(assessment)
    candidate = tmp_path / "import.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, candidate)
    with closing(sqlite3.connect(candidate)) as connection, connection:
        document = json.loads(
            connection.execute(
                "SELECT assessment_json FROM console_response_rule_assessments"
            ).fetchone()[0]
        )
        document["recovery_unicode_witness"] = "証拠 café"
        stored = "[]" if outcome == "malformed" else json.dumps(document)
        if outcome == "invalid_json":
            connection.execute("PRAGMA ignore_check_constraints=ON")
            stored = "{broken"
        connection.execute(
            "UPDATE console_response_rule_assessments SET assessment_json=?", (stored,)
        )
    before = candidate.read_bytes()
    source_before = Path(db.db_path).read_bytes()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("TLDW_", "PYTEST_")):
            env.pop(key)
    home = tmp_path / "cold-home"
    home.mkdir()
    root = Path(__file__).resolve().parents[2]
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(home / "config"),
        XDG_DATA_HOME=str(home / "data"),
        XDG_CACHE_HOME=str(home / "cache"),
        TLDW_TEST_CONFIG_ROOT=str(home / "bootstrap"),
        PYTHONPATH=os.pathsep.join(
            (str(root), str(root / "packages/tldw_profile_core/src"))
        ),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    result = subprocess.run(
        [sys.executable, "-c", _COLD_IMPORT, str(candidate), outcome],
        cwd=root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["constructor_free"] is True
    assert store.list_bindings(binding.scope) == (binding,)
    assert Path(db.db_path).read_bytes() == source_before
    with closing(sqlite3.connect(candidate)) as connection:
        state, revision = connection.execute(
            "SELECT state,binding_revision FROM console_response_rule_bindings"
        ).fetchone()
        assessment_state, encoded = connection.execute(
            "SELECT state,assessment_json FROM console_response_rule_assessments"
        ).fetchone()
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_machine_followup_receipts"
            ).fetchone()[0]
            == 0
        )
    if outcome == "valid":
        assert (state, revision) == ("disabled", binding.binding_revision + 1)
        assert assessment_state == "cancelled"
        document["state"] = "cancelled"
        assert encoded == json.dumps(
            document, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    else:
        assert candidate.read_bytes() == before
        assert (state, revision) == ("enabled", binding.binding_revision)
        assert (assessment_state, encoded) == ("pending", stored)


def test_restricted_import_preparation_refuses_an_unknown_table(rule_store, tmp_path):
    from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

    _, db, _ = rule_store
    candidate = tmp_path / "import.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, candidate)
    with closing(sqlite3.connect(candidate)) as connection, connection:
        connection.execute("CREATE TABLE foreign_payload(value TEXT)")
    with pytest.raises(ValueError, match="unsupported_schema"):
        prepare_imported_candidate(candidate, Event())


@pytest.mark.parametrize("version", [76, 77, 78, 79, 80])
def test_restricted_historical_restore_migrates_only_disposable_copy(
    rule_store, tmp_path, version
):
    import shutil

    from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters

    source = tmp_path / "historical.sqlite"
    with chachanotes_db_at_version(source, version):
        pass
    before = source.read_bytes()
    candidate = tmp_path / "working-import.sqlite"
    shutil.copyfile(source, candidate)
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    assert validate_candidate(owner, candidate, Event(), migrate=True) == ()
    prepare_imported_candidate(candidate, Event())
    with closing(sqlite3.connect(candidate)) as connection:
        assert connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        ).fetchone()[0] == max(owner.schema_policy().versions)
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_response_rule_bindings"
            ).fetchone()[0]
            == 0
        )
    assert source.read_bytes() == before


@pytest.mark.parametrize("version", (80, 81, 82, 83))
@pytest.mark.parametrize("lineage", range(4))
@pytest.mark.parametrize("failure", (None, "deny", "cancel"))
def test_every_accepted_v80_catalog_migrates_to_its_exact_current_variant(
    tmp_path, monkeypatch, lineage, version, failure
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_core_schema import (
        CHACHANOTES_V80_SCHEMAS,
        CHACHANOTES_V81_SCHEMAS,
        CHACHANOTES_V82_SCHEMAS,
        CHACHANOTES_V83_SCHEMAS,
    )

    path = tmp_path / "retained-lineage.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        from tldw_chatbook.DB.canvas_payload_validation import (
            install_canvas_revision_payload_validator,
        )

        install_canvas_revision_payload_validator(connection)
        catalog = {
            80: CHACHANOTES_V80_SCHEMAS,
            81: CHACHANOTES_V81_SCHEMAS,
            82: CHACHANOTES_V82_SCHEMAS,
            83: CHACHANOTES_V83_SCHEMAS,
        }[version][lineage]
        # Virtual tables create their native shadow tables themselves.
        declarations = sorted(
            catalog,
            key=lambda sql: (
                not sql.startswith("CREATE VIRTUAL TABLE"),
                not sql.startswith("CREATE TABLE"),
            ),
        )
        for sql in declarations:
            if sql.startswith("CREATE TABLE sqlite_sequence"):
                continue
            try:
                connection.execute(sql)
            except sqlite3.OperationalError as error:
                if "already exists" not in str(error):
                    raise
                assert connection.execute(
                    "SELECT 1 FROM sqlite_schema WHERE sql=?", (sql,)
                ).fetchone()
        connection.execute(
            "INSERT INTO db_schema_version(schema_name,version) VALUES ('rag_char_chat_schema',?)",
            (version,),
        )
        connection.execute(
            "INSERT INTO conversations(id,root_id,title,client_id) VALUES ('retained','retained','saved lineage','rules')"
        )
    path.chmod(0o600)
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    policy = owner.schema_policy()
    current_version = max(policy.versions)
    current_schemas = tuple(
        schema for version, schema in policy.schema_sql if version == current_version
    )
    from tldw_chatbook.Backup_Recovery import sqlite_validation as validation

    cancel = Event()
    observed = []
    portrait_authority = []
    authorize = validation._Restrictions.authorize

    def observe(self, action, first, second, database, source):
        if self not in observed:
            observed.append(self)
        if action == sqlite3.SQLITE_ALTER_TABLE and self.portrait_migration is not None:
            portrait_authority.append(
                self.canvas_schema
                and not self.changing_schema_trust
                and self.migration_owner == owner.owner_id
            )
        if (
            failure
            and action == sqlite3.SQLITE_ALTER_TABLE
            and second == "character_portrait_performances"
        ):
            if failure == "cancel":
                cancel.set()
            return sqlite3.SQLITE_DENY
        return authorize(self, action, first, second, database, source)

    if version in (82, 83):
        _assert_portrait_migration_authority_is_owner_step_and_source_scoped(version)
    monkeypatch.setattr(validation._Restrictions, "authorize", observe)
    schema_access = validation._canvas_schema_access
    migration_scopes = set()

    @contextmanager
    def checked_schema_access(connection, schema, restrictions=None):
        migrating = (
            restrictions is not None
            and restrictions.migrating
            and restrictions.migration_owner == owner.owner_id
        )
        try:
            with schema_access(connection, schema, restrictions):
                if migrating:
                    assert restrictions.migration_owner == owner.owner_id
                    assert restrictions.migrating
                    assert connection.execute("PRAGMA trusted_schema").fetchone() == (
                        1,
                    )
                    assert connection.execute(
                        "SELECT canvas_revision_payload_valid(?,?,?)",
                        (b"", hashlib.sha256(b"").hexdigest(), 0),
                    ).fetchone() == (1,)
                    with pytest.raises(sqlite3.DatabaseError):
                        connection.execute(
                            "CREATE TABLE unrelated_migration(value TEXT)"
                        )
                    migration_scopes.add(
                        connection.execute(
                            "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
                        ).fetchone()[0]
                    )
                yield
        finally:
            if migrating:
                assert not connection.getconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA)
                assert not restrictions.canvas_schema
                assert not restrictions.changing_schema_trust
                if not restrictions.expired():
                    assert connection.execute("PRAGMA trusted_schema").fetchone() == (
                        0,
                    )
                    with pytest.raises(sqlite3.DatabaseError):
                        connection.execute("PRAGMA trusted_schema=ON")
                    with pytest.raises(sqlite3.DatabaseError):
                        connection.execute(
                            "SELECT canvas_revision_payload_valid(NULL,NULL,NULL)"
                        )

    monkeypatch.setattr(validation, "_canvas_schema_access", checked_schema_access)
    before = path.read_bytes()
    issues = validate_candidate(owner, path, cancel, migrate=True)
    assert portrait_authority and all(portrait_authority)
    for restrictions in observed:
        assert restrictions.portrait_migration is None
        assert not restrictions.migrating
        assert restrictions.migration_owner is None
        assert not restrictions.canvas_schema
        assert not restrictions.changing_schema_trust
    if failure:
        assert issues == (
            "cancelled" if failure == "cancel" else "sqlite_validation_unavailable",
        )
        assert path.read_bytes() == before
        return
    assert issues == ()
    assert migration_scopes == set(range(version, current_version))
    with closing(sqlite3.connect(path)) as connection:
        catalog = tuple(
            r[0]
            for r in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert catalog == current_schemas[lineage]
        assert connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        ).fetchone() == (current_version,)
        assert connection.execute(
            "SELECT title FROM conversations WHERE id='retained'"
        ).fetchone() == ("saved lineage",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("damage", ["future", "hybrid", "changed_index"])
def test_response_rule_candidate_refusal_never_rewrites_rejected_input(
    rule_store, tmp_path, damage
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
    from tldw_chatbook.DB.recovery_core import core_adapters

    _, db, _ = rule_store
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    path = tmp_path / "rejected.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, path)
    with closing(sqlite3.connect(path)) as connection, connection:
        if damage == "future":
            connection.execute(
                "UPDATE db_schema_version SET version=? WHERE schema_name='rag_char_chat_schema'",
                (max(owner.schema_policy().versions) + 1,),
            )
        elif damage == "hybrid":
            connection.execute("DROP INDEX idx_response_rule_drafts_scope")
        else:
            connection.execute("DROP INDEX idx_response_rule_drafts_scope")
            connection.execute(
                "CREATE INDEX idx_response_rule_drafts_scope ON console_response_rule_drafts(scope_id,scope_kind)"
            )
    before = path.read_bytes()
    expected = (
        "unsupported_schema_version" if damage == "future" else "unsupported_schema"
    )
    assert expected in validate_candidate(owner, path, Event(), migrate=True)
    assert path.read_bytes() == before


def _assert_portrait_migration_authority_is_owner_step_and_source_scoped(phase):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

    table = (
        "character_portrait_performances"
        if phase == 82
        else "character_portrait_regions"
    )
    admitted = [
        (sqlite3.SQLITE_CREATE_TABLE, table, None, "main", None),
        (
            sqlite3.SQLITE_CREATE_INDEX,
            f"sqlite_autoindex_{table}_1",
            table,
            "main",
            None,
        ),
        (sqlite3.SQLITE_FUNCTION, None, "json_valid", None, None),
    ]
    if phase == 83:
        admitted.extend(
            [
                (
                    sqlite3.SQLITE_ALTER_TABLE,
                    "main",
                    "character_portrait_performances",
                    None,
                    None,
                ),
                (
                    sqlite3.SQLITE_PRAGMA,
                    "quick_check",
                    "character_portrait_performances",
                    None,
                    None,
                ),
            ]
        )
    denied = [
        (sqlite3.SQLITE_CREATE_TABLE, "unexpected", None, "main", None),
        (sqlite3.SQLITE_ALTER_TABLE, "main", "character_cards", None, None),
        (sqlite3.SQLITE_UPDATE, table, "source_ref_json", "main", None),
        (sqlite3.SQLITE_DELETE, table, None, "main", None),
        (sqlite3.SQLITE_FUNCTION, None, "load_extension", None, None),
    ]
    with closing(sqlite3.connect(":memory:")) as connection:
        restrictions = _Restrictions(connection)
        restrictions.migrating = True
        restrictions.migration_owner = "db.chachanotes.primary"
        restrictions.portrait_migration = phase
        assert all(
            restrictions.authorize(*callback) == sqlite3.SQLITE_OK
            for callback in admitted
        )
        assert all(
            restrictions.authorize(*callback) == sqlite3.SQLITE_DENY
            for callback in denied
        )
        create = admitted[0]
        assert restrictions.authorize(*create[:3], "temp", None) == sqlite3.SQLITE_DENY
        assert (
            restrictions.authorize(*create[:4], "unexpected_trigger")
            == sqlite3.SQLITE_DENY
        )
        restrictions.portrait_migration = 83 if phase == 82 else 82
        assert restrictions.authorize(*create) == sqlite3.SQLITE_DENY
        restrictions.portrait_migration = phase
        restrictions.migration_owner = "db.evals"
        assert restrictions.authorize(*create) == sqlite3.SQLITE_DENY
        restrictions.migration_owner = "db.chachanotes.primary"
        restrictions.migrating = False
        assert restrictions.authorize(*create) == sqlite3.SQLITE_DENY
