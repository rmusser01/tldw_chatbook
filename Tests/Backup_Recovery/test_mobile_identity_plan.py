"""Typed reservations use authenticated native snapshots without owner writes."""

from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


def test_captured_identity_reservations_preserve_domains_and_source_bytes(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=archives,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                plan = reads.identity_plan()
                assert plan == reads.identity_plan()
                assert (
                    plan.mapped("workspace", "workspace-default") == "workspace-default"
                )
                assert (
                    plan.mapped("workspace", "conflicting-workspace")
                    != "conflicting-workspace"
                )
                assert plan.mapped("conversation", "conversation") != "conversation"
                assert plan.mapped("binding", "binding") != "binding"
                assert plan.mapped("conversation", "conversation") != plan.mapped(
                    "workspace", "conflicting-workspace"
                )
                columns, source = reads.page("legacy", "chat", "messages")
                source_ids = {row[columns.index("id")] for row in source}
                columns, canonical = reads.page("canonical", "chat", "messages")
                canonical_ids = {row[columns.index("id")] for row in canonical}
                for identity in source_ids - canonical_ids:
                    assert plan.mapped("message", identity) == identity
                columns, snapshots = reads.page("legacy", "runs", "change_snapshots")
                for row in snapshots:
                    identity = row[columns.index("id")]
                    assert type(identity) is int
                    assert type(plan.mapped("change_snapshot", identity)) is int
                columns, steps = reads.page("legacy", "runs", "agent_run_steps")
                for row in steps:
                    run_id, sequence = (
                        row[columns.index("run_id")],
                        row[columns.index("seq")],
                    )
                    assert plan.mapped("run_step", (run_id, sequence)) == (
                        plan.mapped("run", run_id),
                        sequence,
                    )
                with pytest.raises((ValueError, KeyError), match="identity"):
                    plan.mapped("change_snapshot", "1")
                with pytest.raises(FrozenInstanceError):
                    plan.source_fingerprint = "changed"
            assert _fingerprints(tmp_path / "profile") == before


def test_native_global_order_reservation_uses_allocator_frontier(tmp_path, monkeypatch):
    import sqlite3

    from tldw_chatbook.Backup_Recovery.inventory import discover, read_profile_config
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import legacy_mobile_dir
    from tldw_chatbook.Chat.console_trace_models import _validate_opaque_id

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, config, stores, _, _, _ = values
        source = legacy_mobile_dir(read_profile_config(config)) / "chat.db"
        for path in (source, stores.chat_path):
            connection = sqlite3.connect(path)
            try:
                for sequence in (1, 2) if path == source else (1, 100):
                    connection.execute(
                        "INSERT INTO fleet_progress_messages(sequence,message_id,conversation_id,handle_id,run_id,parent_run_id,agent,body) VALUES (?,?,'conversation','handle','run','parent','agent','private body')",
                        (sequence, "progress-" + str(sequence)),
                    )
                if path != source:
                    connection.execute(
                        "DELETE FROM fleet_progress_messages WHERE sequence=100"
                    )
                connection.commit()
            finally:
                connection.close()
        with session._discovery_reads(private_sqlite=True):
            inventory = discover((config,))
            preview = _preview_selected_mobile(config, config)
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=archives,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                plan = reads.identity_plan()
                assert plan.mapped("fleet_sequence", 1) == 101
                assert plan.mapped("fleet_sequence", 2) == 102
                assert plan.mapped("fleet_message", "progress-2") == "progress-2"
                assert plan.mapped("fleet_message", "progress-1") != "progress-1"
                for item in plan.reservations:
                    if (
                        item.domain.startswith("trace_")
                        and isinstance(item.mapped, str)
                        and item.mapped != item.original
                    ):
                        _validate_opaque_id(item.mapped, item.domain)
            assert _fingerprints(tmp_path / "profile") == before


def test_authenticated_optional_canonical_runs_absence_preserves_run_ids(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(
        tmp_path,
        monkeypatch,
        distinct_selectors=True,
        canonical_runs_absent=True,
        legacy_artifact_root=tmp_path / "profile" / "legacy-artifacts",
    ) as values:
        session, journal, _, stores, preview, inventory, _ = values
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=archives,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads,
        ):
            assert not any(
                purpose == "canonical" and family == "runs"
                for purpose, family, _ in reads.versions
            )
            plan = reads.identity_plan()
            columns, rows = reads.page("legacy", "runs", "agent_runs")
            assert rows
            for row in rows:
                identity = row[columns.index("id")]
                assert plan.mapped("run", identity) == identity
        assert not stores.run_path.exists()


def test_nine_entity_integer_domains_reserve_collisions_without_uuid_coercion(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_workspace_plan import workspace_reads

    with workspace_reads(tmp_path, monkeypatch, entities=True) as reads:
        plan = reads.identity_plan()
        for domain in (
            "keyword",
            "keyword_collection",
            "dictionary",
            "world_book",
            "world_book_entry",
        ):
            assert plan.mapped(domain, 1) == 3
            assert plan.mapped(domain, 2) == 2
            with pytest.raises(ValueError, match="identity_reference_unresolved"):
                plan.mapped(domain, "1")
            with pytest.raises(ValueError, match="identity_reference_unresolved"):
                plan.mapped(domain, True)


def test_citation_composites_remain_typed_pending_native_namespace_proof(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_workspace_plan import workspace_reads

    with workspace_reads(tmp_path, monkeypatch, pending_citation=True) as reads:
        plan = reads.identity_plan()
        pending = [
            item
            for item in plan.pending_dependencies
            if item.table == "rag_citation_traces"
        ]
        assert {(item.handler, item.original) for item in pending} == {
            ("T", ("source-profile", "trace-1")),
            ("T", ("other-profile", "trace-1")),
        }
        with pytest.raises(ValueError, match="identity_reference_unresolved"):
            plan.mapped("citation_trace", ("source-profile", "trace-1"))
        with pytest.raises(ValueError, match="identity_dependencies_pending"):
            plan.require_resolved()


def test_repeated_rule_identity_and_canvas_use_declared_native_domains(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_workspace_plan import workspace_reads

    with workspace_reads(tmp_path, monkeypatch, entities=True) as reads:
        plan = reads.identity_plan()
        assert (
            plan.mapped("canvas", "30000000-0000-4000-8000-000000000001")
            != "30000000-0000-4000-8000-000000000001"
        )
        rule = plan.mapped("response_rule", "same-rule")
        assert rule != "same-rule"
        assert plan.mapped("response_rule_revision", ("same-rule", 1)) == (rule, 1)
        assert plan.mapped("response_rule_revision", ("same-rule", 2)) == (rule, 2)
        assert (
            len([row for row in plan.reservations if row.domain == "response_rule"])
            == 1
        )
