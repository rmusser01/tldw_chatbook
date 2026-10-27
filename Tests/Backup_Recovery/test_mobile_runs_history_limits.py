"""New pure history reads share the native budget, fences and retirement."""

import random

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads
from Tests.Backup_Recovery.test_mobile_sync_plan import _version_default


@pytest.mark.parametrize("fault", ["cancel", "archive", "member", "budget"])
def test_actual_history_failure_preserves_effect_free_retirement(
    tmp_path, monkeypatch, fault
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record
    from tldw_chatbook.Backup_Recovery import (
        mobile_integrated_plan,
        mobile_runs_history,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

    original_seed = mobile_consolidation_fixture._seed

    def native(db, legacy, ids):
        original = ids["artifact_file"]
        parent = original.parent.with_name(ids["parent_run_id"])
        original.parent.rename(parent)
        logfile = parent / original.name
        logfile.rename(parent / "logs.0001.txt")
        generator = random.Random(17)
        for number in range(1, 4 if legacy and fault == "budget" else 2):
            body = (
                generator.randbytes(1_500_000).hex()
                if legacy and fault == "budget"
                else "captured native content"
            )
            raw = encode_record(
                RunLogRecord(
                    number,
                    ids["parent_run_id"],
                    "primary",
                    "model",
                    "original-time",
                    body,
                )
            )
            (parent / f"logs.{number:04d}.txt").write_bytes(raw)

    def seeded(stores, **kwargs):
        return original_seed(stores, **kwargs, runs_seed=native)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    original_integrated = mobile_integrated_plan._plan_integrated
    original_frames = mobile_runs_history._frame_runs
    observed = {}

    def integrated(reads, *, _read_budget=None):
        observed["budget"] = _read_budget
        return original_integrated(reads, _read_budget=_read_budget)

    monkeypatch.setattr(mobile_integrated_plan, "_plan_integrated", integrated)
    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:

        def frames(raw, runs, primary, budget):
            assert budget is observed["budget"]
            result = original_frames(raw, runs, primary, budget)
            if fault == "cancel":
                reads.cancel.set()
            elif fault in {"archive", "member"}:
                path = (
                    reads._source_archive[2].path
                    if fault == "archive"
                    else reads._connections["canonical", "workspaces"][3]
                )
                with path.open("r+b") as stream:
                    original = stream.read(1)
                    stream.seek(0)
                    stream.write(bytes([original[0] ^ 1]))
            return result

        monkeypatch.setattr(mobile_runs_history, "_frame_runs", frames)
        expected = (
            InterruptedError
            if fault == "cancel"
            else _TraceReadLimit
            if fault == "budget"
            else ValueError
        )
        with pytest.raises(expected):
            reads.runs_preflight()
        assert observed["budget"] is not None
    assert not reads.active
    assert reads.roots == []
    assert all(not getattr(r, "close_failed", False) for r in reads.resources)
    with pytest.raises(ValueError, match="inactive"):
        reads.runs_preflight()


def _captured_history_component(reads):
    """Qualify history only, using this scope's actual rows and issued identity."""
    from tldw_chatbook.Backup_Recovery.mobile_identity_plan import _rows
    from tldw_chatbook.Backup_Recovery.mobile_runs_history import _qualify_history
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    identities = reads.identity_plan()
    budget = _TraceReadBudget()
    rows = _rows(reads, "legacy", "runs", "agent_runs", _read_budget=budget)
    budget.container(len(rows))
    runs = {row["id"]: row for row in rows}
    history, blockers = _qualify_history(reads, runs, identities, _read_budget=budget)
    return history, blockers, runs


@pytest.mark.parametrize("tokens", ["spaces", "fields"])
def test_actual_native_header_expansion_is_admitted_before_decoder(
    tmp_path, monkeypatch, tokens
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record
    from tldw_chatbook.Backup_Recovery import mobile_runs_history
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

    seed = mobile_consolidation_fixture._seed

    def native(db, legacy, ids):
        original = ids["artifact_file"]
        parent = original.parent.with_name(ids["parent_run_id"])
        original.parent.rename(parent)
        (parent / original.name).unlink()
        raw = encode_record(
            RunLogRecord(1, ids["parent_run_id"], "primary", "model", "old", "body")
        )
        if legacy:
            header, body = raw.split(b"\n", 1)
            extra = (
                b" " * 500_000
                if tokens == "spaces"
                else b"".join(b" k" + str(n).encode() + b"=v" for n in range(50_000))
            )
            raw = header + extra + b"\n" + body
        (parent / "logs.0001.txt").write_bytes(raw)

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        lambda stores, **kw: seed(stores, **kw, runs_seed=native),
    )
    decoder = mobile_runs_history.decode_record_header
    admit = mobile_runs_history._admit_header
    entered = []
    refused = []

    def admission(raw, start, end, budget):
        try:
            return admit(raw, start, end, budget)
        except _TraceReadLimit:
            refused.append(end - start)
            raise

    def boundary(raw):
        if len(raw) > 100_000:
            entered.append(True)
            raise AssertionError(
                "expanded native header reached decoder before admission"
            )
        return decoder(raw)

    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:
        monkeypatch.setattr(mobile_runs_history, "decode_record_header", boundary)
        monkeypatch.setattr(mobile_runs_history, "_admit_header", admission)
        with pytest.raises(_TraceReadLimit):
            _captured_history_component(reads)
        assert not entered
        assert refused and min(refused) > 100_000
    assert not reads.active and not reads.roots


@pytest.mark.parametrize(
    "case",
    [
        "running-prior",
        "foreign",
        "self",
        "child",
        "self-member",
        "directory",
        "partial",
    ],
)
def test_actual_manifest_proves_only_native_historical_refs_and_segments(
    tmp_path, monkeypatch, case
):
    import json

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record

    seed = mobile_consolidation_fixture._seed
    source = {}

    def native(db, legacy, ids):
        original = ids["artifact_file"]
        parent = original.parent.with_name(ids["parent_run_id"])
        original.parent.rename(parent)
        (parent / original.name).unlink()
        raw = encode_record(
            RunLogRecord(
                1, ids["parent_run_id"], "primary", "model", "old", "parent content"
            )
        )
        (parent / "logs.0001.txt").write_bytes(
            raw + (b"torn" if legacy and case == "partial" else b"")
        )
        # A stale native manifest need not enumerate a survivor's later segment.
        (parent / "logs.0002.txt").write_bytes(b"")
        (parent / "logs.0003.txt").write_bytes(
            encode_record(
                RunLogRecord(
                    1, ids["child_run_id"], "subagent", "model", "old", "child"
                )
            )
        )
        conversation = (
            db.goal_runs.get(ids["goal_id"]).conversation_id
            if case == "foreign"
            else ids["conversation_id"]
        )
        previous = db.create_run(conversation_id=conversation, agent_kind="primary")
        assert db.get_run(previous)["status"] == "running"
        target = (
            ids["parent_run_id"]
            if case == "self"
            else ids["child_run_id"]
            if case == "child"
            else previous
        )
        segment = (
            "MANIFEST"
            if case == "self-member"
            else "logs.0004.txt"
            if case == "directory"
            else "logs.0001.txt"
        )
        if case == "directory":
            (parent / segment).mkdir()
        (parent / "MANIFEST").write_text(
            json.dumps(
                {
                    "run_id": ids["parent_run_id"],
                    "superseded_run_id": target,
                    "segments": [segment],
                }
            )
        )
        if legacy:
            source.update(ids=ids, previous=previous)

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        lambda stores, **kw: seed(stores, **kw, runs_seed=native),
    )
    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=_version_default
    ) as reads:
        history, blockers, runs = _captured_history_component(reads)
        manifests = [p for p in history if p.relative_path.endswith("/MANIFEST")]
        if case == "running-prior":
            assert len(manifests) == 1
            assert manifests[0].run_ids == (source["ids"]["parent_run_id"],)
            assert manifests[0].historical_run_ids == (source["previous"],)
            assert runs[source["previous"]]["status"] == "running"
            empty = next(
                p for p in history if p.relative_path.endswith("logs.0002.txt")
            )
            child = next(
                p for p in history if p.relative_path.endswith("logs.0003.txt")
            )
            assert empty.run_ids == ()
            assert child.run_ids == (source["ids"]["child_run_id"],)
            for proof in (manifests[0], empty, child):
                assert proof.owner_run_id == source["ids"]["parent_run_id"]
                assert proof.owner_association[0] == proof.owner_run_id
        else:
            assert not manifests
            assert any(
                b.table == "agents.history"
                and b.reason
                in {
                    "native_history_owner_unproved",
                    "native_history_missing_member",
                    "native_history_segment_unproved",
                }
                for b in blockers
            )
    assert not reads.active and not reads.roots
