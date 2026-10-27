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
