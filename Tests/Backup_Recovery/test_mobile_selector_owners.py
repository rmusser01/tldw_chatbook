"""Prepared selector object identity is checked at every actual native owner."""

import sqlite3
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepared
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import bind_profile
from tldw_chatbook.Backup_Recovery.inventory import discover
from tldw_chatbook.Backup_Recovery.mobile_consolidation import preview_selected_mobile
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads

OWNERS = ("db.chachanotes.primary", "db.workspaces", "db.agent_runs")


def _seed_no_history(source):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    source.mkdir(parents=True, mode=0o700)
    execution = ConsoleExecutionOwner.acquire(source / "runs.db")
    runs = AgentRunsDB(source / "runs.db", "selector-test", execution_owner=execution)
    chat = CharactersRAGDB(source / "chat.db", "selector-test")
    workspace = WorkspaceDB(source / "workspaces.db", "selector-test")
    try:
        chat.add_conversation({"id": "selector-source", "title": "No file history"})
        for db, version in ((chat, 81), (workspace, 8), (runs, 25)):
            with db.transaction() as cursor:
                cursor.execute("DROP TABLE mobile_consolidation_batches")
                if db is chat:
                    cursor.execute(
                        "UPDATE db_schema_version SET version=81 "
                        "WHERE schema_name='rag_char_chat_schema'"
                    )
                else:
                    cursor.execute("DROP TABLE mobile_import_authority_holds")
                    cursor.execute(
                        "DELETE FROM schema_version WHERE version>?", (version,)
                    )
    finally:
        chat.close_connection()
        workspace.close()
        runs.close()
        execution.close()


@contextmanager
def _selector_case(tmp_path, monkeypatch):
    from Tests.Backup_Recovery import mobile_consolidation_fixture

    monkeypatch.setattr(
        mobile_consolidation_fixture, "seed_legacy_mobile", _seed_no_history
    )
    with prepared(tmp_path, monkeypatch) as values:
        _, journal, source, stores, _, _, authority = values
    target = source.with_name("target.toml")
    candidate = source.with_name("candidate.toml")
    for selector in (target, candidate):
        selector.write_bytes(source.read_bytes())
        selector.chmod(0o600)
        bind_profile(
            authority.control_root.parent,
            selector,
            ("mobile.profile",),
            authority.control_root,
        )
    with _preview_reads():
        inventory = discover((source, target))
    preview = preview_selected_mobile(source, target)
    assert set(preview.capture_scope) == {
        preview.source_dir / leaf for leaf in ("chat.db", "workspaces.db", "runs.db")
    }
    assert {source, target, candidate} <= {
        row.selector for row in preview.selector_provenance
    }
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        session.prepare_mobile_consolidation(
            journal,
            source_config=source,
            target_config=target,
            selections=DiscoverySelections(),
            approved_scope=inventory.scope_digest,
            source_fingerprint=preview.source_fingerprint,
            destination_witnesses=preview.destination_witnesses,
            destination_sidecars=preview.destination_sidecars,
            reviewed_preview=preview,
        )
        yield session, journal, (source, target, candidate), stores


def _replace_same_bytes(selector):
    before = selector.read_bytes()
    identity = selector.stat().st_ino
    selector.rename(selector.with_suffix(".retained"))
    selector.write_bytes(before)
    selector.chmod(0o600)
    assert selector.read_bytes() == before
    assert selector.stat().st_ino != identity


@pytest.mark.parametrize("owner_id", OWNERS)
@pytest.mark.parametrize(
    "selector_index", (0, 1, 2), ids=("source", "target", "candidate")
)
@pytest.mark.parametrize("boundary", ("admission", "transaction"))
def test_same_byte_selector_replacement_refuses_actual_owner(
    tmp_path, monkeypatch, owner_id, selector_index, boundary
):
    with _selector_case(tmp_path, monkeypatch) as (session, journal, selectors, _):
        if boundary == "admission":
            _replace_same_bytes(selectors[selector_index])
            with (
                pytest.raises(bootstrap.RecoveryRequired, match="selector"),
                session.mobile_consolidation_owner(journal, owner_id),
            ):
                pytest.fail("changed selector admitted a native owner")
        else:
            with session.mobile_consolidation_owner(journal, owner_id) as db:
                connection = (
                    db._get_thread_connection()
                    if owner_id == "db.chachanotes.primary"
                    else db._held_connection()
                )
                _replace_same_bytes(selectors[selector_index])
                try:
                    with (
                        pytest.raises(bootstrap.RecoveryRequired, match="selector"),
                        db.transaction(),
                    ):
                        pytest.fail("changed selector admitted a native transaction")
                finally:
                    # Restore the exact reviewed fixture object before owner retirement.
                    selector = selectors[selector_index]
                    selector.rename(selector.with_suffix(".rejected"))
                    selector.with_suffix(".retained").rename(selector)
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


@pytest.mark.parametrize("owner_id", OWNERS)
def test_unchanged_selector_cohort_allows_native_owner_and_physical_retirement(
    tmp_path, monkeypatch, owner_id
):
    with _selector_case(tmp_path, monkeypatch) as (session, journal, _, _):
        with (
            session.mobile_consolidation_owner(journal, owner_id) as db,
            db.transaction() as cursor,
        ):
            assert cursor.execute("SELECT 1").fetchone()[0] == 1
            connection = (
                db._get_thread_connection()
                if owner_id == "db.chachanotes.primary"
                else db._held_connection()
            )
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_new_default_selector_refuses_native_owner(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import capture_service

    candidate = tmp_path / "new-default.toml"
    monkeypatch.setattr(capture_service, "default_config_path", lambda: candidate)
    with _selector_case(tmp_path, monkeypatch) as (session, journal, selectors, _):
        candidate.write_bytes(selectors[0].read_bytes())
        candidate.chmod(0o600)
        with (
            pytest.raises(bootstrap.RecoveryRequired, match="selector"),
            session.mobile_consolidation_owner(journal, "db.chachanotes.primary"),
        ):
            pytest.fail("unreviewed default selector admitted a native owner")
