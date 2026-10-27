"""Actual captured all-owner foundation remains effect-free and noninsertable."""

import hashlib
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads
from Tests.Backup_Recovery.test_mobile_sync_plan import _version_default


@pytest.mark.parametrize("history_case", ["complete", "partial", "foreign", "unknown"])
def test_actual_runs_preflight_preserves_operational_facts_and_native_history(
    tmp_path, monkeypatch, history_case
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
    from tldw_chatbook.DB.console_command_receipts import (
        ConsoleCommandReceipt,
        ConsoleCommandReceiptRepository,
    )

    original_seed = mobile_consolidation_fixture._seed
    source = {}

    def native(db, legacy, ids):
        import json
        from dataclasses import asdict

        from tldw_chatbook.Agents.agent_models import AgentStep

        with db.transaction() as connection:
            connection.execute(
                "UPDATE agent_run_steps SET payload=? WHERE run_id=?",
                (
                    json.dumps(
                        asdict(
                            AgentStep(
                                index=0,
                                kind="tool_result",
                                result="original result",
                                parent_event_id="agent-run:" + ids["parent_run_id"],
                            )
                        )
                    ),
                    ids["child_run_id"],
                ),
            )
        owner = "legacy-owner" if legacy else "canonical-owner"
        db.automatic_work.recover(current_owner_id=owner)
        ids["blob_run_id"] = db.create_run(
            conversation_id=ids["conversation_id"], agent_kind="primary"
        )
        db.set_status(ids["blob_run_id"], "done", result=b"body\x00\xff")
        receipt = ConsoleCommandReceipt(
            command_id="command-same",
            server_epoch="historical-epoch",
            console_id="historical-console",
            action="subagent_stop",
            target=ConsoleTarget(
                server_epoch="historical-epoch",
                console_id="historical-console",
                session_id="opaque-session",
                incarnation_id="opaque-incarnation",
                run_id=ids["child_run_id"],
            ),
            payload_fingerprint="a" * 64,
            state="pending",
            issued_ns=10,
            expires_ns=20,
            native_id=ids["child_run_id"],
        )
        ConsoleCommandReceiptRepository(db).claim(receipt)
        raw = b"".join(
            encode_record(
                RunLogRecord(
                    number=index,
                    run_id=run,
                    kind=kind,
                    type="model",
                    ts="2026-10-25T00:00:00Z",
                    content="Unchanged " + run,
                )
            )
            for index, run, kind in (
                (1, ids["parent_run_id"], "primary"),
                (2, ids["child_run_id"], "subagent"),
            )
        )
        original = ids["artifact_file"]
        parent = original.parent.with_name(ids["parent_run_id"])
        original.parent.rename(parent)
        old = parent / original.name
        logfile = old.with_name("logs.000001.txt")
        old.rename(logfile)
        if legacy and history_case == "partial":
            raw += b"torn unframed tail"
        elif legacy and history_case == "foreign":
            raw += encode_record(
                RunLogRecord(
                    number=3,
                    run_id="foreign-run",
                    kind="primary",
                    type="model",
                    ts="old",
                    content="foreign",
                )
            )
        elif legacy and history_case == "unknown":
            logfile = logfile.with_name("unknown.dat")
        logfile.write_bytes(raw)
        (parent.parent / ".gitignore").write_text("*\n")
        if legacy:
            source.update(ids=ids, raw=raw, receipt=receipt, owner=owner)

    def seeded(stores, **kwargs):
        return original_seed(stores, **kwargs, runs_seed=native)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:
        build = getattr(reads, "runs_preflight", None)
        assert callable(build), "actual captured scope needs fixed all-owner preflight"
        plan = build()
        assert all(
            proof.relative_path not in repr(plan) for proof in plan.history_observations
        )
        from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES

        assert plan.field_ownership is plan.integrated.field_ownership
        ownership = {
            (family, table, handler): columns
            for family, table, columns, handler in plan.field_ownership
        }
        run_specs = [spec for spec in _TABLES if spec.handler == "A-runs"]
        assert (
            len(run_specs) == 22 and sum(len(spec.columns) for spec in run_specs) == 212
        )
        assert all(
            ownership[spec.family, spec.table, spec.handler] == spec.columns
            for spec in run_specs
        )
        assert any(
            duty.table == "goal_runs"
            and duty.requirement
            == "native_goal_current_binding_authority_and_fresh_result_review"
            for duty in plan.pending_duties
        )
        ids = plan.identities
        original = source["ids"]
        rows = {(r.table, r.original): dict(r.values) for r in plan.rows}
        child = rows["agent_runs", (original["child_run_id"],)]
        assert child["result"] == "saved result"
        assert (
            rows["agent_runs", (original["blob_run_id"],)]["result"] == b"body\x00\xff"
        )
        assert child["status"] == "done"
        assert child["wake_delivered_at"] is None
        assert child["parent_run_id"] == ids.mapped("run", original["parent_run_id"])
        for reservation in plan.integrated.identities.reservations:
            assert reservation in ids.reservations
        for row in plan.rows:
            if row.table == "goal_runs":
                goal = dict(row.values)
                chain = next(
                    dict(c.values)
                    for c in plan.rows
                    if c.table == "automatic_work_chains"
                    and dict(c.values)["id"] == goal["chain_id"]
                )
                assert chain["root_submission_id"] == "goal:" + goal["id"]
        facts = {f.table: f for f in plan.operational_history}
        assert dict(facts["automatic_work_runtime_owner"].values) == {
            "singleton": 1,
            "owner_id": "legacy-owner",
        }
        receipt = facts["console_command_receipts"]
        values = dict(receipt.values)
        assert values["command_id"] == "command-same"
        assert values["state"] == "pending"
        assert values["native_id"] == original["child_run_id"]
        assert values["issued_ns"] == 10 and values["expires_ns"] == 20
        assert (
            "run",
            original["child_run_id"],
            ids.mapped("run", original["child_run_id"]),
        ) in receipt.associations
        assert not any(
            a[0] in {"session", "epoch", "native_id"} for a in receipt.associations
        )
        columns, values = reads.page(
            "canonical", "runs", "automatic_work_runtime_owner"
        )
        assert (
            dict(zip(columns, values[0], strict=True))["owner_id"] == "canonical-owner"
        )
        proofs = [
            p
            for p in plan.history
            if p.sha256 == hashlib.sha256(source["raw"]).hexdigest()
        ]
        if history_case == "complete":
            assert len(proofs) == 1
            assert set(proofs[0].run_ids) == {
                original["parent_run_id"],
                original["child_run_id"],
            }
        else:
            assert not proofs
            reason = {
                "partial": "native_history_partial",
                "foreign": "native_history_foreign_run",
                "unknown": "native_history_format_unproved",
            }[history_case]
            assert any(
                b.table == "agents.history" and b.reason == reason
                for b in plan.blockers
            )
        assert any(
            d.requirement == "native_passive_command_history_retention"
            and d.original == ("command-same",)
            for d in plan.pending_duties
        )
        with pytest.raises(ValueError, match="pending"):
            replace(plan, pending_duties=()).require_complete()
        with pytest.raises(ValueError, match="pending"):
            plan.integrated.require_complete()
    with pytest.raises(ValueError, match="inactive"):
        build()
    if history_case == "complete":
        from Tests.DB.test_mobile_import_holds import hold
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        database = AgentRunsDB(":memory:")
        try:
            allowed = {spec.table for spec in run_specs}
            with database.transaction() as connection:
                connection.execute("PRAGMA defer_foreign_keys=ON")
                # Native run-chain trigger is immediate even with deferred FKs.
                # This disposable sink exercises native consumers, not a writer API.
                ordered = sorted(
                    plan.rows, key=lambda r: r.table != "automatic_work_chains"
                )
                for row in ordered:
                    if row.table not in allowed:
                        continue
                    values = dict(row.values)
                    columns = tuple(values)
                    connection.execute(
                        "INSERT INTO "
                        + row.table
                        + " ("
                        + ",".join(columns)
                        + ") VALUES ("
                        + ",".join("?" for _ in columns)
                        + ")",
                        tuple(values.values()),
                    )
            child_id = ids.mapped("run", original["child_run_id"])
            native_child = database.get_run(child_id)
            assert native_child["result"] == "saved result"
            assert native_child["steps"][0]["result"] == "original result"
            assert native_child["steps"][0][
                "parent_event_id"
            ] == "agent-run:" + ids.mapped("run", original["parent_run_id"])
            hold(database, ("run", child_id))
            assert database.get_run(child_id) == native_child
            assert all(
                r["id"] != child_id
                for r in database.pending_wake_results(native_child["conversation_id"])
            )
            goal_row = next(dict(r.values) for r in plan.rows if r.table == "goal_runs")
            native_goal = database.goal_runs.get(goal_row["id"])
            assert native_goal.payload_hash == goal_row["payload_hash"]
        finally:
            database.close()


def test_native_run_projection_requires_all_of_its_retained_steps(
    tmp_path, monkeypatch
):
    import json

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
    from tldw_chatbook.DB.console_command_receipts import (
        ConsoleCommandReceipt,
        ConsoleCommandReceiptRepository,
    )

    original_seed = mobile_consolidation_fixture._seed
    source = {}

    def native(db, legacy, ids):
        original = ids["artifact_file"]
        parent = original.parent.with_name(ids["parent_run_id"])
        original.parent.rename(parent)
        logfile = parent / "logs.0001.txt"
        (parent / original.name).rename(logfile)
        logfile.write_bytes(
            encode_record(
                RunLogRecord(
                    1, ids["child_run_id"], "subagent", "model", "old", "retained child"
                )
            )
        )
        (parent / "MANIFEST").write_text(
            json.dumps({"run_id": ids["parent_run_id"], "segments": ["logs.0001.txt"]})
        )
        (parent / "logs.0002.txt").write_bytes(b"")
        ConsoleCommandReceiptRepository(db).claim(
            ConsoleCommandReceipt(
                command_id="lost-child-command",
                server_epoch="old",
                console_id="old",
                action="subagent_stop",
                target=ConsoleTarget(
                    server_epoch="old",
                    console_id="old",
                    session_id="opaque",
                    incarnation_id="opaque",
                    run_id=ids["child_run_id"],
                ),
                payload_fingerprint="a" * 64,
                state="pending",
                issued_ns=10,
                expires_ns=20,
                native_id=ids["child_run_id"],
            )
        )
        if legacy:
            source.update(ids)

    def seeded(stores, **kwargs):
        return original_seed(stores, **kwargs, runs_seed=native)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:
        plan = reads.runs_preflight()
        assert not any(
            r.table == "agent_runs" and r.original == (source["child_run_id"],)
            for r in plan.rows
        )
        assert any(
            b.table == "agent_runs"
            and b.original == (source["child_run_id"],)
            and b.reason == "native_run_step_input_lost"
            for b in plan.blockers
        )
        assert any(
            r.table == "agent_runs" and r.original == (source["parent_run_id"],)
            for r in plan.rows
        )

        assert not any(source["child_run_id"] in p.run_ids for p in plan.history)
        assert not any(p.relative_path.endswith("/MANIFEST") for p in plan.history)
        empty = next(
            p for p in plan.history if p.relative_path.endswith("logs.0002.txt")
        )
        assert empty.run_ids == () and empty.owner_run_id == source["parent_run_id"]
        assert any(
            b.reason == "native_history_segment_dependency_blocked"
            for b in plan.blockers
        )
        assert any(
            source["child_run_id"] in p.run_ids
            and not p.associations
            and p.owner_association is None
            for p in plan.history_observations
        )
        receipt = next(
            h for h in plan.operational_history if h.table == "console_command_receipts"
        )
        assert dict(receipt.values)["native_id"] == source["child_run_id"]
        assert receipt.associations == ()
        assert any(
            b.reason == "native_operational_run_dependency_blocked"
            and b.original == ("lost-child-command",)
            for b in plan.blockers
        )
        assert any(
            b.reason == "native_history_run_dependency_blocked" for b in plan.blockers
        )
        accepted = {(r.table, r.original) for r in plan.rows}
        assert all(
            not d.original
            or d.table in {"automatic_work_runtime_owner", "console_command_receipts"}
            or (d.table, d.original) in accepted
            for d in plan.pending_duties
        )
        assert any(
            d.requirement == "native_passive_command_history_retention"
            for d in plan.pending_duties
        )
        assert any(not d.original and d.table == "*" for d in plan.pending_duties)


def test_actual_runs24_preflight_preserves_authenticated_receipt_absence(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_source_runs24 import _source_layout

    _source_layout(monkeypatch, 24)
    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:
        plan = reads.runs_preflight()
        assert ("legacy", "runs", 24) in reads.versions
        assert not any(
            f.table == "console_command_receipts" for f in plan.operational_history
        )
        assert not any(
            d.table == "console_command_receipts" for d in plan.pending_duties
        )
        assert any(r.table == "agent_runs" for r in plan.source_rows)
        with pytest.raises(ValueError, match="mobile_sqlite_table_invalid"):
            reads.page("legacy", "runs", "console_command_receipts")
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()
    assert not reads.active
