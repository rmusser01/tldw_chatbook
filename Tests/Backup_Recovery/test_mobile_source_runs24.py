"""Actual pre-feature runs24 capture preserves its one native table absence."""

import errno
import hashlib
import os
import sqlite3
from contextlib import closing

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads
from Tests.Backup_Recovery.test_mobile_sync_plan import _version_default


def _source_layout(monkeypatch, version, *, receipt=False, malformed=None):
    from Tests.Backup_Recovery import mobile_consolidation_fixture as fixture
    from tldw_chatbook.DB.console_command_receipts import COMMAND_RECEIPT_MIGRATION

    original = fixture._seed
    evidence = {}

    def seeded(stores, **kwargs):
        result = original(stores, **kwargs, legacy_runs_version=version)
        if kwargs["legacy"]:
            # Fixture construction ends here, before preview/admission/capture.
            with closing(sqlite3.connect(stores.run_path)) as connection:
                if malformed == "unexpected24":
                    connection.execute(COMMAND_RECEIPT_MIGRATION[0])
                elif malformed == "missing25":
                    connection.execute("DROP TABLE console_command_receipts")
                if receipt:
                    connection.execute(
                        "INSERT INTO console_command_receipts VALUES ('retained-command','epoch','console','send','{}','fingerprint','applied',1,2,'native','kept','then')"
                    )
                connection.commit()
                evidence["catalog"] = tuple(
                    connection.execute(
                        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
                    )
                )
                evidence["versions"] = tuple(
                    connection.execute(
                        "SELECT version FROM schema_version ORDER BY version"
                    )
                )
            evidence["path"] = stores.run_path
            evidence["sha256"] = hashlib.sha256(
                stores.run_path.read_bytes()
            ).hexdigest()
        return result

    monkeypatch.setattr(fixture, "_seed", seeded)
    return evidence


@pytest.mark.parametrize("version", (24, 25))
def test_actual_source_version_capture_identity_and_integrated_plan(
    tmp_path, monkeypatch, version
):
    from tldw_chatbook.Backup_Recovery import crypto, mobile_captured_sqlite
    from tldw_chatbook.Backup_Recovery.mobile_identity_plan import _rows

    original = _source_layout(monkeypatch, version, receipt=version == 25)
    retire = crypto._CryptoJob.retire
    jobs = []

    def retired(job):
        result = retire(job)
        if type(job.scope) is mobile_captured_sqlite._CapturedSQLite:
            assert job.process.poll() is not None
            assert all(not worker.is_alive() for worker in job.workers)
            assert all(
                pipe.closed
                for pipe in (job.process.stdin, job.process.stdout, job.process.stderr)
            )
            jobs.append(job)
        return result

    monkeypatch.setattr(crypto._CryptoJob, "retire", retired)
    with content_reads(
        tmp_path,
        monkeypatch,
        lambda connection, purpose: None,
        native_seed=_version_default,
    ) as reads:
        assert reads.versions == (
            ("legacy", "chat", 81),
            ("legacy", "workspaces", 8),
            ("legacy", "runs", version),
            ("canonical", "chat", 82),
            ("canonical", "workspaces", 9),
            ("canonical", "runs", 26),
        )
        plan = reads.integrated_plan()
        connection = reads._connections["legacy", "runs"][0]
        assert (
            tuple(
                connection.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
                )
            )
            == original["catalog"]
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT version FROM schema_version ORDER BY version"
                )
            )
            == original["versions"]
        )
        catalog = reads.schema_catalog()
        assert any(
            p == "canonical" and f == "runs" and t == "console_command_receipts"
            for p, f, t, _ in catalog
        )
        assert any(
            p == "legacy" and f == "runs" and t == "console_command_receipts"
            for p, f, t, _ in catalog
        ) == (version == 25)
        manifest = plan.manifest
        assert {(r.purpose, r.family, r.table): r.columns for r in manifest.tables} == {
            (p, f, t): c for p, f, t, c in catalog
        }
        identities = plan.identities
        command_rows = _rows(reads, "legacy", "runs", "console_command_receipts")
        commands = [r for r in identities.reservations if r.domain == "command"]
        if version == 24:
            assert command_rows == () and commands == []
            with pytest.raises(ValueError, match="^mobile_sqlite_table_invalid$"):
                reads.page("legacy", "runs", "console_command_receipts")
        else:
            assert len(command_rows) == len(commands) == 1
            assert (
                command_rows[0]["command_id"]
                == commands[0].original
                == "retained-command"
            )
        assert plan.pending_handlers == ("A-runs", "owner-write", "public-cutover")
        assert any(d.table == "agent_runs" for d in plan.pending_duties)
        assert any(
            d.table == "console_command_receipts" for d in plan.pending_duties
        ) == (version == 25)
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()
        connections = tuple(value[0] for value in reads._connections.values())
        pins = tuple(
            (
                member[1].capture_stream.fileno(),
                os.fstat(member[1].capture_stream.fileno()),
            )
            for member in reads._connections.values()
        )
        assert len(connections) == len(pins) == 6
        assert (
            hashlib.sha256(original["path"].read_bytes()).hexdigest()
            == original["sha256"]
        )
    assert (
        len(jobs) == 4 and not reads.active and not reads.resources and not reads.roots
    )
    assert not tuple((tmp_path / "archives").glob(".mobile-read-*"))
    for fd, before in pins:
        try:
            after = os.fstat(fd)
        except OSError as error:
            assert error.errno == errno.EBADF
        else:
            assert (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    with pytest.raises(ValueError, match="inactive"):
        _rows(reads, "legacy", "runs", "console_command_receipts")
    assert (
        hashlib.sha256(original["path"].read_bytes()).hexdigest() == original["sha256"]
    )


@pytest.mark.parametrize("version,malformed", ((24, "unexpected24"), (25, "missing25")))
def test_native_catalog_rejects_wrong_version_receipt_presence(
    tmp_path, monkeypatch, version, malformed
):
    from threading import Event

    from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepared
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    original = _source_layout(monkeypatch, version, malformed=malformed)
    with prepared(tmp_path, monkeypatch) as values:
        _, _, _, _, preview, inventory, _ = values
        assert "unsupported_schema_version" in preview.issues
        assert not inventory.complete
        owner = next(a for a in recovery_adapters() if a.owner_id == "db.agent_runs")
        assert validate_candidate(owner, original["path"], Event(), migrate=False) == (
            "unsupported_schema_version",
        )
    assert (
        hashlib.sha256(original["path"].read_bytes()).hexdigest() == original["sha256"]
    )


@pytest.mark.parametrize("version,declared", ((24, 25), (25, 24)))
def test_native_observed_version_refuses_mismatched_archive_declaration(
    tmp_path, monkeypatch, version, declared
):
    from tldw_chatbook.Backup_Recovery import mobile_captured_sqlite, sqlite_validation

    original = _source_layout(monkeypatch, version)
    members, check = mobile_captured_sqlite._members, sqlite_validation._check
    seen, scopes = [], []

    def wrong_declaration(scope, partition, document):
        if partition.purpose == "legacy":
            scopes.append(scope)
            # Negative boundary input only: native observed bytes remain untouched.
            document = document.model_copy(
                update={
                    "owners": tuple(
                        row.model_copy(update={"schema_version": declared})
                        if row.owner_id == "db.agent_runs"
                        else row
                        for row in document.owners
                    )
                }
            )
        return members(scope, partition, document)

    def observed(connection, owner, policy, restrictions):
        result = check(connection, owner, policy, restrictions)
        if owner.owner_id == "db.agent_runs" and result == ((), version):
            seen.append(result)
        return result

    monkeypatch.setattr(mobile_captured_sqlite, "_members", wrong_declaration)
    monkeypatch.setattr(sqlite_validation, "_check", observed)
    with (
        pytest.raises(ValueError, match="^mobile_sqlite_schema_changed$"),
        content_reads(
            tmp_path,
            monkeypatch,
            lambda connection, purpose: None,
            native_seed=_version_default,
        ),
    ):
        pytest.fail("declared/observed mismatch admitted")
    assert seen and scopes
    for scope in scopes:
        assert not scope.active and not scope.resources and not scope.roots
        for connection, *_ in scope._connections.values():
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
    assert not tuple((tmp_path / "archives").glob(".mobile-read-*"))
    assert (
        hashlib.sha256(original["path"].read_bytes()).hexdigest() == original["sha256"]
    )
