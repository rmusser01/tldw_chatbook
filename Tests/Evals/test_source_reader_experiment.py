"""Fixture runner contracts; all provider work uses local recording doubles."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "Helper_Scripts/Benchmarks/library_source_reader.py"


def cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_offline_prepare_report_freezes_heldout_matrix_and_preserves_existing_output(
    tmp_path,
):
    prepared, report = tmp_path / "prepared", tmp_path / "report"
    result = cli("prepare", "--output", prepared, "--include-followup")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((prepared / "manifest.json").read_text())
    heldout = [row for row in manifest["matrix"] if row["split"] == "heldout"]
    assert len(heldout) == 72
    assert len({row["case_id"] for row in heldout}) == 12
    assert {row["repeat"] for row in heldout} == {1, 2}
    assert (
        len(
            {
                row["case_id"]
                for row in manifest["matrix"]
                if row["split"] == "development"
            }
        )
        == 4
    )
    assert (
        len(
            {
                row["case_id"]
                for row in manifest["matrix"]
                if row["scenario"] == "followup"
            }
        )
        == 3
    )
    assert (prepared / "library.sqlite").is_file()
    assert all(case["essential_facts"] for case in manifest["cases"])
    frozen = (prepared / "manifest.json").read_bytes()
    assert cli("prepare", "--output", prepared).returncode != 0
    assert (prepared / "manifest.json").read_bytes() == frozen
    result = cli("report", "--prepared", prepared, "--output", report)
    assert result.returncode == 0, result.stderr
    document = json.loads((report / "report.json").read_text())
    assert document["primary"]["decision"] == "inconclusive"
    assert document["followup"]["decision"] == "inconclusive"
    assert cli("report", "--prepared", prepared, "--output", report).returncode != 0


def test_live_run_requires_explicit_models_endpoint_and_all_caps(tmp_path):
    result = cli("run", "--prepared", tmp_path, "--output", tmp_path / "run")
    assert result.returncode == 2
    for required in (
        "--main-model",
        "--worker-model",
        "--spend-cap-usd",
        "--request-cap",
        "--token-cap",
    ):
        assert required in result.stderr
    assert not (tmp_path / "run").exists()


def run_args(tmp_path):
    from argparse import Namespace

    return Namespace(
        prepared=tmp_path / "prepared",
        output=tmp_path / "run",
        main_model="gpt-4o",
        worker_model="gpt-4o-mini",
        base_url="https://example.invalid/v1",
        api_key_env="FIXTURE_EXPERIMENT_KEY",
        request_cap=100,
        token_cap=3000000,
        spend_cap_usd=100.0,
        context_limit=32768,
        input_limit=24000,
        output_limit=512,
        split="development",
    )


class RecordingGateway:
    def __init__(self, *, unknown_usage=False):
        self.requests = []
        self.unknown_usage = unknown_usage

    async def complete_auxiliary(self, request):
        from tldw_chatbook.Chat.console_provider_gateway import (
            AuxiliaryCompletionResult,
        )
        from tldw_chatbook.Chat.provider_usage import ProviderUsage

        self.requests.append(request)
        payload = json.loads(request.messages[1]["content"])
        text = "Answer from supplied fixture evidence."
        if "packets" in payload:
            packet = payload["packets"][0]
            text = json.dumps(
                {
                    "findings": [
                        {
                            "statement": "Candidate fixture fact",
                            "evidence": [
                                {
                                    "packet_id": packet["packet_id"],
                                    "quote": packet["text"].split(".")[0] + ".",
                                }
                            ],
                        }
                    ]
                }
            )
        usage = (
            None
            if self.unknown_usage
            else ProviderUsage(
                uncached_input=100,
                output=20,
                provider=request.resolution.provider,
                model=request.resolution.model,
            )
        )
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=request.resolution.model,
            text=text,
            usage=usage,
        )


@pytest.mark.asyncio
async def test_dry_run_writes_reviewable_configuration_without_dispatch(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    args = run_args(tmp_path)
    args.dry_run = True
    prepare(args.prepared)
    monkeypatch.setenv(args.api_key_env, "key")
    gateway = RecordingGateway()
    assert await run_experiment(args, gateway=gateway) == []
    assert gateway.requests == []
    manifest = json.loads((args.output / "run_manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["reserved_requests"] == 24


@pytest.mark.asyncio
async def test_run_records_exact_evidence_worker_plus_main_cost_and_no_credentials(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    args = run_args(tmp_path)
    prepare(args.prepared)
    monkeypatch.setenv(args.api_key_env, "secret-not-an-artifact")
    gateway = RecordingGateway()
    attempts = await run_experiment(args, gateway=gateway)
    assert len(gateway.requests) == 24
    reader = [row for row in attempts if row["arm"] == "reader"]
    assert len(reader) == 8
    assert all(row["status"] == "ok" and len(row["calls"]) == 2 for row in reader)
    assert all(row["cost_usd"] == pytest.approx(0.000477) for row in reader)
    assert all(
        row["status"] == "index_unavailable"
        for row in attempts
        if row["arm"] == "retrieval"
    )
    artifacts = "".join(path.read_text() for path in args.output.glob("*.json"))
    assert "secret-not-an-artifact" not in artifacts
    assert "practice garden opens at 09:00" in artifacts
    blind = json.loads((args.output / "blind_grading_packet.json").read_text())
    key = json.loads((args.output / "grading_key.json").read_text())
    assert len(blind) == len(key) == 24
    assert all(
        "arm" not in row and "model" not in row and "worker" not in row for row in blind
    )
    assert all(
        row["original_selected_sources"] and row["success"] is None for row in blind
    )
    assert {key[row["review_id"]]["arm"] for row in blind} == {
        "direct",
        "reader",
        "retrieval",
    }
    assert all(
        request.resolution.reasoning_effort is None for request in gateway.requests
    )
    assert all(request.sensitive is True for request in gateway.requests)
    assert all(
        set(message) == {"role", "content"}
        for request in gateway.requests
        for message in request.messages
    )
    assert all(
        request.resolution.base_url == "https://example.invalid/v1"
        for request in gateway.requests
    )


@pytest.mark.asyncio
async def test_unknown_usage_stops_all_later_calls_and_incomplete_matrix_is_retained(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    args = run_args(tmp_path)
    prepare(args.prepared)
    monkeypatch.setenv(args.api_key_env, "key")
    gateway = RecordingGateway(unknown_usage=True)
    attempts = await run_experiment(args, gateway=gateway)
    assert len(gateway.requests) == 1
    assert len(attempts) == 24
    assert any(
        row["status"] == "unknown_usage" and row["cost_usd"] is None for row in attempts
    )
    assert any(row["status"] == "stopped" for row in attempts)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_tokens,output_tokens", [(100, 20), (100, 0), (0, 20)])
async def test_deepseek_rate_boundary_keeps_bounded_cost_separate_from_unknown_usage(
    tmp_path, monkeypatch, input_tokens, output_tokens
):
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Evals.source_reader import experiment

    class BoundaryClock(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            moment = datetime(2026, 9, 8, 3, 59, 59, tzinfo=UTC) + timedelta(
                seconds=2 * cls.calls
            )
            cls.calls += 1
            return moment.astimezone(tz)

    class DeepSeekGateway(RecordingGateway):
        async def complete_auxiliary(self, request):
            result = await super().complete_auxiliary(request)
            return replace(
                result,
                usage=replace(
                    result.usage, uncached_input=input_tokens, output=output_tokens
                ),
            )

    args = run_args(tmp_path)
    args.provider = "deepseek"
    args.main_model, args.worker_model = "deepseek-v4-pro", "deepseek-v4-flash"
    experiment.prepare(args.prepared)
    monkeypatch.setenv(args.api_key_env, "key")
    monkeypatch.setattr(experiment, "datetime", BoundaryClock)
    gateway = DeepSeekGateway()
    attempts = await experiment.run_experiment(args, gateway=gateway)
    first = next(row for row in attempts if row["calls"])
    call = first["calls"][0]
    assert call["pricing"]["tier"] == "mixed"
    assert call["cost_usd"] is None
    assert not call.get("known_cost_usd")
    if input_tokens and output_tokens:
        assert call["cost_upper_bound_usd"] > 0
        assert len(gateway.requests) == 24
        assert all(
            row["status"] == "ok" for row in attempts if row["arm"] != "retrieval"
        )
        assert "response" in call
    else:
        assert call["cost_upper_bound_usd"] is None
        assert len(gateway.requests) == 1
        assert first["status"] == "unknown_usage"
        assert "response" not in call


@pytest.mark.asyncio
async def test_complete_run_preflight_rejects_tiny_budget_before_any_dispatch(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    args = run_args(tmp_path)
    prepare(args.prepared)
    args.spend_cap_usd = 0.00001
    monkeypatch.setenv(args.api_key_env, "key")
    gateway = RecordingGateway()
    with pytest.raises(ValueError, match="run_budget_exceeded"):
        await run_experiment(args, gateway=gateway)
    assert gateway.requests == []
    assert not args.output.exists()


@pytest.mark.asyncio
async def test_retrieval_scopes_before_search_and_rejects_foreign_or_stale_results():
    from types import SimpleNamespace

    from tldw_chatbook.Evals.source_reader.experiment import ScopedRetrievalAdapter
    from tldw_chatbook.Evals.source_reader.sources import Source

    class Index:
        def __init__(self):
            self.calls = []
            self.source_id = "7"
            self.revision = "2"

        async def search(self, question, **kwargs):
            self.calls.append(kwargs)
            return [
                SimpleNamespace(
                    document="South entrance is step-free.",
                    metadata={
                        "source_type": "media",
                        "source_id": self.source_id,
                        "revision": self.revision,
                    },
                )
            ]

    index = Index()
    source = Source(
        "public-seven", "2", "Access", "South entrance is step-free. North has steps."
    )
    adapter = ScopedRetrievalAdapter(index, {"public-seven": "7"}, {"7": "2"})
    with pytest.raises(ValueError, match="invalid_scope"):
        await adapter.retrieve("Entrance?", ())
    assert index.calls == []
    result = await adapter.retrieve("Entrance?", (source,))
    assert result["evidence"][0]["start"] == 0
    assert result["evidence"][0]["quote"] == "South entrance is step-free."
    assert result["query_embedding_cost_usd"] is None
    assert index.calls[0]["metadata_allowlist"] == [
        {"source_type": {"media"}, "source_id": {"7"}}
    ]
    assert "filter_metadata" not in index.calls[0]
    index.source_id = "8"
    with pytest.raises(ValueError, match="retrieval_scope_violation"):
        await adapter.retrieve("Entrance?", (source,))
    index.source_id, index.revision = "7", "1"
    with pytest.raises(ValueError, match="retrieval_revision_changed"):
        await adapter.retrieve("Entrance?", (source,))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_tokens,output_tokens,known_cost",
    [(100, 20, 0.000027), (100, 0, 0.000015), (0, 20, 0.000012)],
)
async def test_cancelled_call_drains_late_usage_without_late_answer_and_preserves_matrix(
    tmp_path, monkeypatch, input_tokens, output_tokens, known_cost
):
    import asyncio
    from dataclasses import replace

    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    args = run_args(tmp_path)
    prepare(args.prepared)
    monkeypatch.setenv(args.api_key_env, "key")
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowGateway(RecordingGateway):
        async def complete_auxiliary(self, request):
            entered.set()
            await release.wait()
            result = await super().complete_auxiliary(request)
            return replace(
                result,
                usage=replace(
                    result.usage, uncached_input=input_tokens, output=output_tokens
                ),
            )

    runner = asyncio.create_task(run_experiment(args, gateway=SlowGateway()))
    await entered.wait()
    runner.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await runner
    attempts = json.loads((args.output / "attempts.json").read_text())
    assert len(attempts) == 24
    cancelled = next(row for row in attempts if row["status"] == "cancelled")
    if input_tokens and output_tokens:
        assert cancelled["cost_usd"] == pytest.approx(known_cost)
    else:
        assert cancelled["cost_usd"] is None
        assert cancelled["calls"][0]["cost_usd"] is None
        late = json.loads((args.output / "late_usage.json").read_text())
        assert late["cost_usd"] is None
        assert late["known_cost_usd"] == pytest.approx(known_cost)
    assert cancelled["known_cost_usd"] == pytest.approx(known_cost)
    assert cancelled["calls"][0]["usage"]["uncached_input"] == input_tokens
    assert "response" not in cancelled["calls"][0]
    assert "answer" not in cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_tokens,status",
    [(3000, "usage_protocol_deviation"), (0, "unknown_usage"), (-1, "unknown_usage")],
)
async def test_usage_above_output_allowance_stops_further_dispatch(
    tmp_path, monkeypatch, output_tokens, status
):
    from dataclasses import replace

    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    args = run_args(tmp_path)
    prepare(args.prepared)
    monkeypatch.setenv(args.api_key_env, "key")

    class ExcessOutput(RecordingGateway):
        async def complete_auxiliary(self, request):
            result = await super().complete_auxiliary(request)
            return replace(
                result,
                usage=replace(
                    result.usage,
                    output=max(output_tokens, 0) if output_tokens != -1 else 20,
                    uncached_input=0 if output_tokens == -1 else 100,
                ),
            )

    gateway = ExcessOutput()
    attempts = await run_experiment(args, gateway=gateway)
    assert len(gateway.requests) == 1
    assert any(row["status"] == status for row in attempts)


def test_report_retains_partial_and_extra_observed_spend(tmp_path):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, report

    prepared, run = tmp_path / "prepared", tmp_path / "run"
    manifest = prepare(prepared)
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"manifest_sha256": manifest["manifest_sha256"]})
    )
    attempts = [
        {
            "case_id": "held-01",
            "repeat": 1,
            "arm": "reader",
            "status": "unknown_usage",
            "cost_usd": None,
            "known_cost_usd": 0.000027,
            "calls": [{"cost_usd": 0.000027}, {"cost_usd": None}],
        },
        {
            "case_id": "extra",
            "repeat": 1,
            "arm": "direct",
            "status": "ok",
            "cost_usd": 0.3,
        },
    ]
    (run / "attempts.json").write_text(json.dumps(attempts))
    result = report(prepared, tmp_path / "report", run_dir=run)
    assert result["accounting"]["known_total_usd"] == pytest.approx(0.300027)
    assert result["unassigned_attempts"] == [attempts[1]]
    assert "unexpected_case_rows" in result["primary"]["reasons"]
    assert result["primary"]["decision"] == "inconclusive"


def test_report_rejects_grader_reducing_the_frozen_essential_fact_total(tmp_path):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, report

    prepared = tmp_path / "prepared"
    prepare(prepared)
    grades = tmp_path / "grades.json"
    grades.write_text(
        json.dumps(
            [
                {
                    "case_id": "held-01",
                    "repeat": 1,
                    "arm": "reader",
                    "success": True,
                    "critical_error": False,
                    "essential_correct": 1,
                    "essential_total": 1,
                }
            ]
        )
    )
    result = report(prepared, tmp_path / "report", grades_path=grades)
    assert "grade_essential_total_mismatch" in result["primary"]["reasons"]
    assert result["primary"]["decision"] == "inconclusive"
