"""Retained native notes keep missing snapshot identity without allocator authority."""

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads
from Tests.Backup_Recovery.test_mobile_sync_plan import _version_default


@pytest.mark.parametrize("case", ["retained", "base_collision", "mismatch"])
def test_retained_note_snapshot_union_keeps_native_missing_history(
    tmp_path, monkeypatch, case
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture

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
                            AgentStep(index=0, kind="tool_result", result="retained")
                        )
                    ),
                    ids["child_run_id"],
                ),
            )
        snapshot = db.change_snapshots_for_run(ids["child_run_id"])[0]
        note = db.add_change_note(
            run_id=ids["child_run_id"],
            root=snapshot["root"],
            path="original.txt",
            hunk_index=0,
            hunk_header="@@ original @@",
            hunk_excerpt="retained exact excerpt",
            note="retained review",
            snapshot_id=snapshot["id"],
        )
        null_note = db.add_change_note(
            run_id=ids["child_run_id"],
            root=snapshot["root"],
            path="legacy.txt",
            hunk_index=0,
            hunk_header="@@ legacy @@",
            hunk_excerpt="legacy excerpt",
            note="legacy NULL",
        )
        assert db.delete_change_snapshots_older_than("9999-01-01") == 1
        db.record_change_snapshot(
            run_id=ids["child_run_id"],
            root=snapshot["root"],
            baseline_sha="later-baseline",
            end_sha="later-end",
        )
        current = db.change_snapshots_for_run(ids["child_run_id"])[0]
        if not legacy and case == "base_collision":
            db.add_change_note(
                run_id=ids["child_run_id"],
                root=snapshot["root"],
                path="second.txt",
                hunk_index=0,
                hunk_header="@@ second @@",
                hunk_excerpt="canonical retained",
                note="second",
                snapshot_id=current["id"],
            )
            db.delete_change_snapshots_older_than("9999-01-01")
            db.record_change_snapshot(
                run_id=ids["child_run_id"],
                root=snapshot["root"],
                baseline_sha="third",
                end_sha="third-end",
            )
        if legacy and case == "mismatch":
            bad_note = db.add_change_note(
                run_id=ids["parent_run_id"],
                root=snapshot["root"],
                path="wrong-owner.txt",
                hunk_index=0,
                hunk_header="@@ wrong @@",
                hunk_excerpt="retained wrong-owner",
                note="uncertain",
                snapshot_id=current["id"],
            )
            source["bad_note"] = bad_note
        if legacy:
            source["live_snapshot"] = current["id"]
            source.update(
                note=note,
                null_note=null_note,
                snapshot=snapshot["id"],
                run=ids["child_run_id"],
            )

    def seeded(stores, **kwargs):
        return original_seed(stores, **kwargs, runs_seed=native)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:
        plan = reads.runs_preflight()
        rows = {(r.table, r.original): dict(r.values) for r in plan.rows}
        assert ("change_notes", (source["note"],)) in rows
        note = rows["change_notes", (source["note"],)]
        assert note["snapshot_id"] == plan.identities.mapped(
            "change_snapshot", source["snapshot"]
        )
        assert note["snapshot_id"] not in {1, 2, 3}
        assert note["hunk_excerpt"] == "retained exact excerpt"
        assert note["delivered_at"] is None
        assert rows["change_notes", (source["null_note"],)]["snapshot_id"] is None
        assert not any(
            r.table == "change_snapshots"
            and dict(r.values)["id"] == note["snapshot_id"]
            for r in plan.rows
        )
        assert all(
            r in plan.identities.reservations
            for r in plan.integrated.identities.reservations
        )
        assert plan.snapshot_allocator.source_sequence == 2
        assert plan.snapshot_allocator.canonical_sequence == (
            3 if case == "base_collision" else 2
        )
        if case == "base_collision":
            assert (
                plan.integrated.identities.mapped(
                    "change_snapshot", source["live_snapshot"]
                )
                == source["live_snapshot"]
            )
            assert any(
                b.reason == "native_snapshot_historical_collision"
                and b.original == (source["live_snapshot"],)
                for b in plan.blockers
            )
        if case == "mismatch":
            assert ("change_notes", (source["bad_note"],)) not in rows
            assert any(
                b.reason == "native_snapshot_owner_unproved"
                and b.original == (source["bad_note"],)
                for b in plan.blockers
            )
        assert plan.snapshot_allocator.required_frontier >= note["snapshot_id"]
        assert any(
            p.original_note == source["note"]
            and p.original_snapshot == source["snapshot"]
            and p.mapped_snapshot == note["snapshot_id"]
            and p.missing
            for p in plan.note_provenance
        )
        assert any(
            d.requirement == "native_snapshot_allocator_admission_before_effects"
            for d in plan.pending_duties
        )
