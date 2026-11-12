"""Every native captured table and column has an explicit owning disposition."""

import pytest

from Tests.Backup_Recovery.test_mobile_workspace_plan import workspace_reads


def test_fixed_dispositions_cover_native_schemas_and_refuse_pending_handlers(
    tmp_path, monkeypatch
):
    with workspace_reads(tmp_path, monkeypatch) as reads:
        catalog = reads.schema_catalog()
        plan = reads.plan_manifest()
        observed = {
            (row.purpose, row.family, row.table): row.columns for row in plan.tables
        }
        assert observed == {
            (purpose, family, table): columns
            for purpose, family, table, columns in catalog
        }
        for (purpose, family), (
            connection,
            _,
            tables,
            _,
            _,
        ) in reads._connections.items():
            assert {
                table for p, f, table in observed if (p, f) == (purpose, family)
            } == tables
            for table in tables:
                # Independent SQLite schema metadata, never synthesized reader rows.
                cursor = connection.execute(
                    'SELECT * FROM "' + table.replace('"', '""') + '" LIMIT 0'
                )
                try:
                    columns = tuple(row[0] for row in cursor.description)
                finally:
                    cursor.close()
                assert observed[purpose, family, table] == columns
        from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import _legacy_absent
        from tldw_chatbook.Backup_Recovery.mobile_identity_plan import (
            _CITATION_IDENTITIES,
            _COMPOSITES,
            _DOMAINS,
        )

        for domain in _DOMAINS:
            if ("legacy", domain.family, domain.table) not in observed:
                assert _legacy_absent(reads, "legacy", domain.family, domain.table)
                continue
            assert domain.column in observed["legacy", domain.family, domain.table]
        for _, family, table, fields in _COMPOSITES:
            assert all(
                column in observed["legacy", family, table] for column, _ in fields
            )
        for table, fields in _CITATION_IDENTITIES:
            assert all(
                column in observed["legacy", "chat", table] for column, _ in fields
            )
        assert plan.pending_handlers == ("C", "T", "S", "A-runs")
        with pytest.raises(ValueError, match="required_handlers_pending"):
            plan.require_complete()
        from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _disposition

        columns, _ = reads.page("legacy", "workspaces", "workspace_records", limit=1)
        with pytest.raises(ValueError, match="column_disposition_missing"):
            _disposition(
                "workspaces",
                "workspace_records",
                columns + ("future_user_setting",),
                False,
            )
        assert (
            _disposition(
                "chat",
                "conversations",
                observed["legacy", "chat", "conversations"],
                False,
                reads=reads,
            ).handler
            == "C"
        )
        assert (
            _disposition(
                "chat", "sync_log", observed["legacy", "chat", "sync_log"], False
            ).handler
            == "S"
        )
        assert (
            _disposition(
                "chat",
                "console_trace_events",
                observed["legacy", "chat", "console_trace_events"],
                False,
            ).handler
            == "T"
        )


def test_canonical_workflow_is_preserved_without_importing_its_statistics(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture as fixture
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result

    original_seed = fixture._seed
    saved = []

    def seed_runs(db, legacy, _metadata):
        assert not legacy
        saved.append(saved_result(db, tmp_path)[-1])
        with db.connection() as connection:
            connection.execute("ANALYZE")

    def seed(stores, *, legacy, **kwargs):
        return original_seed(
            stores, legacy=legacy, runs_seed=None if legacy else seed_runs, **kwargs
        )

    monkeypatch.setattr(fixture, "_seed", seed)
    with workspace_reads(tmp_path, monkeypatch) as reads:
        plan = reads.plan_manifest()
        assert not any(row.table.startswith("sqlite_stat") for row in plan.tables)
        workflow = [row for row in plan.tables if row.table.startswith("workflow_")]
        assert len(workflow) == 15
        assert all(
            (row.purpose, row.family, row.handler)
            == ("canonical", "runs", "native-preserved")
            for row in workflow
        )
        _connection, restrictions, *_ = reads._connections["canonical", "runs"]
        assert "sqlite_stat1" in restrictions.workflow_statistics
        columns, results = reads.page("canonical", "runs", "workflow_results")
        assert len(results) == len(saved) == 1
        assert results[0][columns.index("payload_json")] == saved[0].canonical_json()
        with pytest.raises(ValueError, match="mobile_sqlite_table_invalid"):
            reads.page("legacy", "runs", "workflow_results")
        with pytest.raises(ValueError, match="mobile_sqlite_table_invalid"):
            reads.page("canonical", "runs", "sqlite_stat1")


def test_catalog_refuses_copied_retired_and_substituted_native_reads(
    tmp_path, monkeypatch
):
    from copy import copy

    with workspace_reads(tmp_path, monkeypatch) as reads:
        with pytest.raises(ValueError, match="mobile_sqlite_inactive"):
            copy(reads).schema_catalog()
        member = next(iter(reads._connections.values()))
        candidate = member[3]
        saved = candidate.with_name(candidate.name + ".retained")
        candidate.rename(saved)
        candidate.write_bytes(saved.read_bytes())
        candidate.chmod(0o600)
        try:
            with pytest.raises(ValueError, match="mobile_sqlite_candidate_changed"):
                reads.schema_catalog()
        finally:
            candidate.unlink()
            saved.rename(candidate)
        assert reads.schema_catalog()
    with pytest.raises(ValueError, match="mobile_sqlite_inactive"):
        reads.schema_catalog()


@pytest.mark.parametrize("exhaustion", ("deadline", "steps", "cancel"))
def test_catalog_preserves_actual_native_resource_limits(
    tmp_path, monkeypatch, exhaustion
):
    from time import monotonic

    from tldw_chatbook.Backup_Recovery import sqlite_validation

    with workspace_reads(tmp_path, monkeypatch) as reads:
        restrictions = next(iter(reads._connections.values()))[1]
        if exhaustion == "deadline":
            restrictions.deadline = monotonic() - 1
        elif exhaustion == "steps":
            restrictions.steps = sqlite_validation._STEP_BUDGET + 1
        else:
            reads.cancel.set()
        with pytest.raises(
            (ValueError, InterruptedError), match="sqlite_resource_limit|cancelled"
        ):
            reads.schema_catalog()


def test_populated_standalone_study_has_body_free_exact_prewrite_blocker(
    tmp_path, monkeypatch
):
    with workspace_reads(tmp_path, monkeypatch, unsupported_study=True) as reads:
        plan = reads.plan_manifest()
        assert len(plan.unsupported_records) == 1
        record = plan.unsupported_records[0]
        assert record.table == "topics" and record.identity == ("standalone-topic",)
        assert "private" not in repr(record)
        with pytest.raises(ValueError, match="unsupported"):
            plan.require_complete()


@pytest.mark.parametrize("plan_kind", ("manifest", "content"))
def test_plan_return_rechecks_candidate_mutated_after_catalog(
    tmp_path, monkeypatch, plan_kind
):
    """Pure metadata comparisons never replace the final physical proof."""
    from tldw_chatbook.Backup_Recovery import mobile_content_graph, mobile_plan_manifest

    with workspace_reads(tmp_path, monkeypatch) as reads:
        candidate = reads._connections["canonical", "workspaces"][3]
        original_bytes = candidate.read_bytes()
        mutated = []

        def alter_candidate():
            if not mutated:
                with candidate.open("ab") as stream:
                    stream.write(b"post-catalog-candidate-mutation")
                mutated.append(True)

        if plan_kind == "manifest":
            original = mobile_plan_manifest._disposition

            def disposition(*args, **kwargs):
                result = original(*args, **kwargs)
                alter_candidate()
                return result

            monkeypatch.setattr(mobile_plan_manifest, "_disposition", disposition)
            operation = reads.plan_manifest
        else:
            original = mobile_content_graph._GraphBuilder.close_dependencies

            def close_dependencies(builder):
                result = original(builder)
                alter_candidate()
                return result

            monkeypatch.setattr(
                mobile_content_graph._GraphBuilder,
                "close_dependencies",
                close_dependencies,
            )
            operation = lambda: mobile_content_graph._plan_content_graph(reads)
        try:
            with pytest.raises(ValueError, match="mobile_sqlite_candidate_changed"):
                operation()
            assert mutated == [True]
        finally:
            # Restore this disposable fault-injection member for native retirement.
            candidate.write_bytes(original_bytes)


def test_installed_schema_catalog_exports_keep_canonical_identity():
    from typing import get_type_hints

    from tldw_chatbook.Backup_Recovery import mobile_identity_plan, mobile_plan_manifest
    from tldw_chatbook.DB import mobile_consolidation_receipts, mobile_import_authority

    assert mobile_plan_manifest._TableSpec is mobile_import_authority._TableSpec
    assert mobile_plan_manifest._TABLES is mobile_import_authority._TABLES
    assert mobile_identity_plan._Domain is mobile_import_authority._Domain
    assert mobile_identity_plan._DOMAINS is mobile_import_authority._DOMAINS
    assert mobile_identity_plan._COMPOSITES is mobile_import_authority._COMPOSITES
    assert mobile_consolidation_receipts._TABLES is mobile_import_authority._TABLES
    assert mobile_consolidation_receipts._DOMAINS is mobile_import_authority._DOMAINS
    assert (
        mobile_consolidation_receipts._COMPOSITES is mobile_import_authority._COMPOSITES
    )
    assert get_type_hints(mobile_plan_manifest._TableSpec)["columns"] == tuple[str, ...]
    assert get_type_hints(mobile_identity_plan._Domain)["scalar"] is type
