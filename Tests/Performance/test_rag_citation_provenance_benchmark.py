"""Contracts for the network-free RAG citation provenance benchmark."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    REPO_ROOT / "Helper_Scripts" / "Benchmarks" / "rag_citation_provenance_benchmark.py"
)
FIXTURE_ROOT = REPO_ROOT / "Tests" / "fixtures" / "rag_citation_provenance"
MANIFEST_PATH = FIXTURE_ROOT / "manifest_v1.json"
CORPUS_PATH = FIXTURE_ROOT / "corpus_v1.json"
BASELINE_PATH = (
    REPO_ROOT / "Docs" / "Development" / "RAG" / "citation-provenance-baseline-v1.json"
)
MISSING = object()


def _load_benchmark() -> ModuleType:
    assert BENCHMARK_PATH.is_file(), f"missing benchmark runner: {BENCHMARK_PATH}"
    spec = importlib.util.spec_from_file_location(
        "rag_citation_provenance_benchmark", BENCHMARK_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict:
    assert path.is_file(), f"missing fixture: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_ids(value):
    if isinstance(value, dict):
        if "id" in value:
            yield value["id"]
        for child in value.values():
            yield from _walk_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_ids(child)


def _replace_nested(document: dict, path: tuple[str, ...], value: object) -> None:
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    if value is MISSING:
        parent.pop(path[-1])
    else:
        parent[path[-1]] = value


def test_manifest_references_versioned_files_with_matching_digests() -> None:
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["schema_version"] == 1
    assert manifest["fixture_version"] == "rag-citation-provenance-v1"
    for record in manifest["files"]:
        path = FIXTURE_ROOT / record["path"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert len(payload) == record["bytes"]
        assert len(payload.decode("utf-8")) == record["characters"]


def test_corpus_has_deterministic_ids_stable_counts_and_required_shapes() -> None:
    manifest = _load_json(MANIFEST_PATH)
    corpus = _load_json(CORPUS_PATH)
    ids = list(_walk_ids(corpus))

    assert corpus["schema_version"] == 1
    assert corpus["fixture_version"] == manifest["fixture_version"]
    assert ids
    assert len(ids) == len(set(ids))
    assert all(item.startswith("ragcp-v1-") for item in ids)
    assert {source["kind"] for source in corpus["sources"]} == {
        "media",
        "note",
        "conversation",
    }
    assert {
        shape["submitted_evidence_count"] for shape in corpus["evidence_shapes"]
    } == {
        1,
        8,
        32,
        64,
    }
    assert manifest["answer_cardinalities"] == {
        answer["id"]: answer["citation_occurrences"] for answer in corpus["answers"]
    }
    assert manifest["evidence_cardinalities"] == {
        shape["id"]: shape["submitted_evidence_count"]
        for shape in corpus["evidence_shapes"]
    }


def test_corpus_covers_semantic_storage_and_legacy_cases() -> None:
    corpus = _load_json(CORPUS_PATH)

    assert {
        "unicode",
        "repeated_markers",
        "grouped_markers",
        "repaired_answer",
    } <= {answer["case"] for answer in corpus["answers"]}
    assert {"embedded", "server_reference", "ephemeral", "redacted"} == {
        item["storage_mode"] for item in corpus["storage_cases"]
    }
    assert {"EvidenceBundle", "CitationRef", "chat_rag_context_sidecar"} == {
        item["record_type"] for item in corpus["legacy_records"]
    }
    assert any(
        shape["cited_evidence_count"] < shape["submitted_evidence_count"]
        for shape in corpus["evidence_shapes"]
    )


def test_every_frozen_limit_has_exact_and_one_unit_over_cases() -> None:
    benchmark = _load_benchmark()
    corpus = _load_json(CORPUS_PATH)

    grouped: dict[str, list[dict]] = {}
    for case in corpus["boundary_cases"]:
        grouped.setdefault(case["limit_name"], []).append(case)

    assert set(grouped) == set(benchmark.LIMITS)
    for name, limit in benchmark.LIMITS.items():
        cases = grouped[name]
        assert {case["expected"] for case in cases} == {"accept", "reject"}
        exact = next(case for case in cases if case["expected"] == "accept")
        over = next(case for case in cases if case["expected"] == "reject")

        assert benchmark.validate_boundary_case(exact) == limit
        with pytest.raises(ValueError, match=name):
            benchmark.validate_boundary_case(over)


def test_governed_trace_boundary_uses_individually_valid_snapshots() -> None:
    benchmark = _load_benchmark()
    corpus = _load_json(CORPUS_PATH)
    governed_cases = [
        case
        for case in corpus["boundary_cases"]
        if case["limit_name"] == "governed_trace_bytes"
    ]

    for case in governed_cases:
        workload = benchmark.materialize_boundary(case)
        snapshots = workload["governed_snapshots"]
        assert sum(len(item.encode("utf-8")) for item in snapshots) == case["units"]
        assert max(len(item.encode("utf-8")) for item in snapshots) <= 64 * 1024


def test_boundary_materializers_are_domain_shaped() -> None:
    benchmark = _load_benchmark()
    corpus = _load_json(CORPUS_PATH)

    materialized = {
        case["limit_name"]: benchmark.materialize_boundary(case)
        for case in corpus["boundary_cases"]
        if case["expected"] == "accept"
    }

    assert isinstance(materialized["aggregate_json_bytes"]["trace"], dict)
    assert "snapshot_text" in materialized["snapshot_utf8_bytes"]
    assert "prompt_sets" in materialized["prompt_sets"]
    assert "evidence" in materialized["evidence_per_prompt"]["prompt_set"]
    assert "answer_attempts" in materialized["answer_attempts"]
    assert "occurrences" in materialized["citation_occurrences"]["selected_answer"]
    assert "candidates" in materialized["retrieval_candidates_per_run"]["run"]
    assert isinstance(materialized["locator_json_bytes"]["locator"], dict)
    assert isinstance(materialized["observation_json_bytes"]["observation"], dict)
    assert "legacy_messages" in materialized["migration_batch_messages"]


def test_generation_workload_cycles_genuine_single_evidence_corpus_cases() -> None:
    benchmark = _load_benchmark()
    corpus = _load_json(CORPUS_PATH)

    workload = benchmark._workload_from_corpus(corpus)
    cases = workload["generation_cases"]

    assert len(cases) == len(corpus["sources"]) * len(corpus["answers"])
    assert {case["source_id"] for case in cases} == {
        source["id"] for source in corpus["sources"]
    }
    assert {case["answer_case"] for case in cases} == {
        answer["case"] for answer in corpus["answers"]
    }
    assert all(case["evidence_count"] == 1 for case in cases)
    for case in cases:
        assert (
            sum(source["text"] in case["prompt"] for source in corpus["sources"]) == 1
        )


@pytest.mark.asyncio
async def test_deterministic_gateway_applies_first_token_latency_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_benchmark()
    observed_delays: list[float] = []

    async def record_delay(delay: float) -> None:
        observed_delays.append(delay)

    monkeypatch.setattr(benchmark.asyncio, "sleep", record_delay)
    gateway = benchmark._DeterministicGateway("Corpus answer")

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            SimpleNamespace(),
            [{"role": "user", "content": "Corpus prompt"}],
        )
    ]

    assert observed_delays == [benchmark.MOCK_FIRST_TOKEN_DELAY_SECONDS]
    assert "".join(chunks) == "Corpus answer"


def test_runner_defaults_to_five_warmups_and_at_least_thirty_samples() -> None:
    benchmark = _load_benchmark()

    args = benchmark.parse_args([])

    assert args.warmups == 5
    assert args.samples >= 30
    assert args.mode == "baseline"


def test_sample_group_uses_isolated_temp_chachanotes_db_and_sidecar(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()

    with benchmark.sample_group_workspace(tmp_path, "inspector") as workspace:
        assert workspace.db_path.is_file()
        assert workspace.sidecar_path.is_file()
        assert workspace.db_path.is_relative_to(tmp_path)
        assert workspace.sidecar_path.is_relative_to(tmp_path)
        assert workspace.db_path.name == "ChaChaNotes.db"
        assert workspace.sidecar_path.name == "chat_rag_context.json"


def test_benchmark_host_state_context_isolates_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    original = {
        "HOME": "sentinel-home",
        "XDG_CONFIG_HOME": "sentinel-config",
        "XDG_DATA_HOME": "sentinel-data",
        "TLDW_CONFIG_PATH": "sentinel-config.toml",
        "TLDW_TEST_MODE": "sentinel-test-mode",
        "OPENAI_API_KEY": "sentinel-secret",
    }
    for key, value in original.items():
        monkeypatch.setenv(key, value)

    with benchmark.isolated_benchmark_host_state(tmp_path / "benchmark-host"):
        assert os.environ["TLDW_TEST_MODE"] == "1"
        assert Path(os.environ["HOME"]).is_relative_to(tmp_path)
        assert Path(os.environ["XDG_CONFIG_HOME"]).is_relative_to(tmp_path)
        assert Path(os.environ["XDG_DATA_HOME"]).is_relative_to(tmp_path)
        assert Path(os.environ["TLDW_CONFIG_PATH"]).is_relative_to(tmp_path)
        assert "OPENAI_API_KEY" not in os.environ

    assert {key: os.environ.get(key) for key in original} == original


def test_cli_never_reads_or_writes_host_config_data_or_secrets(tmp_path: Path) -> None:
    host_home = tmp_path / "sentinel-host"
    host_config = host_home / ".config" / "tldw_cli" / "config.toml"
    host_data = host_home / ".local" / "share" / "tldw_cli"
    host_config.parent.mkdir(parents=True)
    host_data.mkdir(parents=True)
    secret = "sk-sentinel-host-secret-never-load"
    host_config.write_text(
        "\n".join(
            [
                "[general]",
                'users_name = "sentinel_host_user"',
                "[paths]",
                f"data_dir = {json.dumps(str(host_data))}",
                "[API]",
                f"openai_api_key = {json.dumps(secret)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    marker = host_data / "do-not-touch.txt"
    marker.write_text("sentinel host data\n", encoding="utf-8")

    def host_snapshot() -> dict[Path, bytes | None]:
        return {
            path.relative_to(host_home): path.read_bytes() if path.is_file() else None
            for path in host_home.rglob("*")
        }

    before = host_snapshot()

    process_temp = tmp_path / "runner-temp"
    process_temp.mkdir()
    output = process_temp / "baseline.json"
    environment = {
        **os.environ,
        "HOME": str(host_home),
        "XDG_CONFIG_HOME": str(host_config.parent.parent),
        "XDG_DATA_HOME": str(host_data.parent.parent),
        "TLDW_CONFIG_PATH": str(host_config),
        "OPENAI_API_KEY": secret,
        "TMPDIR": str(process_temp),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    environment.pop("PYTEST_CURRENT_TEST", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_PATH),
            "--mode",
            "baseline",
            "--samples",
            "1",
            "--warmups",
            "0",
            "--output",
            output.name,
        ],
        cwd=process_temp,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    emitted = completed.stdout + completed.stderr
    assert secret not in emitted
    assert str(host_home) not in emitted
    assert str(REPO_ROOT) not in emitted
    assert str(process_temp) not in emitted
    assert "host-state" not in emitted
    assert completed.stdout == f"Wrote {output.name}\n"
    assert completed.stderr == ""
    assert host_snapshot() == before
    assert output.is_relative_to(process_temp)
    assert _load_json(output)["budgets"]["overall_pass"] is True


@pytest.mark.parametrize(
    "extra",
    [
        ["--base-url", "https://example.invalid/v1"],
        ["--provider", "openai"],
    ],
)
def test_baseline_rejects_external_url_or_provider(extra: list[str]) -> None:
    benchmark = _load_benchmark()
    args = benchmark.parse_args(["--mode", "baseline", *extra])

    with pytest.raises(ValueError, match="network-free|mock-local"):
        benchmark.validate_args(args)


@pytest.mark.parametrize("mode", ["baseline", "qualification"])
@pytest.mark.parametrize(
    "extra",
    [
        ["--external-target", "https://example.invalid/health"],
        ["--external-timeout-seconds", "1"],
        ["--provider", "external-http-v1"],
        ["--base-url", "https://example.invalid/v1"],
    ],
)
def test_local_modes_reject_external_only_arguments(
    mode: str,
    extra: list[str],
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    argv = ["--mode", mode, *extra]
    if mode == "qualification":
        baseline = tmp_path / "baseline.json"
        baseline.write_text("{}\n", encoding="utf-8")
        argv.extend(["--baseline", str(baseline)])

    with pytest.raises(ValueError, match="external|network-free|mock-local"):
        benchmark.validate_args(benchmark.parse_args(argv))


def test_summary_reports_median_and_p95_never_minimum() -> None:
    benchmark = _load_benchmark()

    result = benchmark.summarize([1.0, 2.0, 3.0, 100.0])

    assert result == {"median": 2.5, "p95": 100.0}
    assert "minimum" not in result


@pytest.mark.asyncio
async def test_baseline_uses_the_measured_current_path_as_its_unchanged_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    observations = iter((1.0, 2.0, 3.0, 4.0))

    async def measured_once(_db_path: Path) -> float:
        return next(observations)

    monkeypatch.setattr(benchmark, "_measure_console_ttfb_once", measured_once)
    result = await benchmark._measure_first_token(
        tmp_path / "ChaChaNotes.db",
        samples=2,
        warmups=0,
    )

    assert result["candidate_ms"] == result["control_ms"]
    assert result["regression_vs_control"] == {
        "milliseconds": 0.0,
        "percent": 0.0,
    }


@pytest.mark.asyncio
async def test_prefeature_qualification_uses_the_same_no_provenance_control_series(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    observations = iter((1.0, 2.0, 3.0, 4.0))

    async def measured_once(_db_path: Path) -> float:
        return next(observations)

    monkeypatch.setattr(benchmark, "_measure_console_ttfb_once", measured_once)
    result = await benchmark._measure_first_token(
        tmp_path / "ChaChaNotes.db",
        samples=2,
        warmups=0,
        qualification=True,
    )

    assert result["candidate_ms"] == result["control_ms"]
    assert result["regression_vs_control"]["percent"] == 0.0


def test_repository_storage_candidate_uses_real_sealed_write_and_summary_read(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()

    result = benchmark._measure_repository_storage(
        tmp_path / "repository.sqlite",
        samples=2,
        warmups=1,
        snapshots=("evidence" * 512,),
    )

    assert set(result["message_only_control_ms"]) == {"median", "p95"}
    assert set(result["sealed_write_ms"]) == {"median", "p95"}
    assert set(result["summary_read_ms"]) == {"median", "p95"}
    assert result["persisted_trace_rows"] == 3
    assert result["persisted_owner_rows"] == 3
    assert result["control_path"].endswith("citation_write=None)")
    assert result["candidate_path"].endswith("citation_write=sealed)")


def test_migration_candidate_uses_real_service_and_restart_is_duplicate_free(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    workspace = benchmark.SampleWorkspace(
        tmp_path,
        tmp_path / "migration.sqlite",
        tmp_path / "chat_rag_context.json",
    )

    result = benchmark._measure_migration(
        workspace,
        samples=1,
        warmups=0,
        legacy_records=_load_json(CORPUS_PATH)["legacy_records"],
    )

    assert result["candidate_path"].endswith(
        "CitationLegacyMigrationService.migrate_next_batch"
    )
    assert result["control_path"] == "bounded legacy fixture scan"
    assert result["persisted_trace_rows"] == 100
    assert result["persisted_owner_rows"] == 100
    assert result["duplicate_proxy_rows_after_restart"]["p95"] == 0
    assert result["messages_per_second"]["median"] >= 100


@pytest.mark.asyncio
async def test_repository_storage_candidate_runs_only_in_qualification_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    calls: list[Path] = []
    candidate = {
        "message_only_control_ms": {"median": 1.0, "p95": 1.0},
        "sealed_write_ms": {"median": 2.0, "p95": 2.0},
        "summary_read_ms": {"median": 1.0, "p95": 1.0},
        "persisted_trace_rows": 1,
        "persisted_owner_rows": 1,
        "control_path": "ChatPersistenceService.create_message(citation_write=None)",
        "candidate_path": "ChatPersistenceService.create_message(citation_write=sealed)",
    }

    def measured(
        db_path: Path,
        *,
        samples: int,
        warmups: int,
        snapshots,
    ):
        del samples, warmups, snapshots
        calls.append(db_path)
        return candidate

    monkeypatch.setattr(benchmark, "_measure_repository_storage", measured)
    baseline_result = await benchmark.run_benchmark(
        mode="baseline",
        samples=1,
        warmups=0,
        scratch_root=tmp_path / "baseline",
    )
    assert calls == []
    assert "repository_storage" not in baseline_result["metrics"]

    environment = benchmark.environment_metadata()
    baseline = {
        "fixture_version": benchmark.FIXTURE_VERSION,
        "environment": environment,
        "metrics": benchmark.empty_passing_metrics(),
    }
    qualification_result = await benchmark.run_benchmark(
        mode="qualification",
        samples=1,
        warmups=0,
        scratch_root=tmp_path / "qualification",
        baseline=baseline,
    )

    assert len(calls) == 1
    assert qualification_result["metrics"]["repository_storage"] == candidate
    assert qualification_result["budgets"]["checks"]["repository_storage"]["pass"]


def test_budget_schema_covers_all_six_families() -> None:
    benchmark = _load_benchmark()

    assert set(benchmark.BUDGETS) == {
        "first_token",
        "finalization",
        "inspector_load",
        "trace_size",
        "database_growth",
        "migration",
    }
    assert benchmark.BUDGETS["first_token"]["max_regression_percent"] == 10
    assert benchmark.BUDGETS["first_token"]["max_regression_ms"] == 25
    assert benchmark.BUDGETS["migration"]["minimum_messages_per_second"] == 100


def test_qualification_requires_present_compatible_committed_baseline(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()

    absent = benchmark.parse_args(["--mode", "qualification"])
    with pytest.raises(ValueError, match="--baseline"):
        benchmark.validate_args(absent)

    incompatible_path = tmp_path / "baseline.json"
    incompatible_path.write_text(
        json.dumps(
            {
                "fixture_version": "wrong",
                "environment": {"result_schema_version": 999},
            }
        ),
        encoding="utf-8",
    )
    incompatible = benchmark.parse_args(
        [
            "--mode",
            "qualification",
            "--baseline",
            str(incompatible_path),
        ]
    )
    benchmark.validate_args(incompatible)
    compatibility = benchmark.check_baseline_compatibility(
        json.loads(incompatible_path.read_text(encoding="utf-8")),
        benchmark.environment_metadata(),
    )

    assert compatibility["compatible"] is False
    assert compatibility["reasons"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metrics", "finalization", "standard_ms", "p95"), MISSING),
        (("metrics", "first_token", "candidate_ms", "p95"), "slow"),
        (("metrics", "database_growth", "bytes", "p95"), True),
        (("metrics", "inspector_load", "cold_ms", "p95"), float("nan")),
        (
            ("metrics", "migration", "messages_per_second", "median"),
            float("inf"),
        ),
        (("metrics", "trace_size", "aggregate_json_bytes"), float("-inf")),
        (("metrics", "first_token", "candidate_ms", "p95"), 0.0),
        (("metrics", "database_growth", "governed_bytes_per_answer"), -1),
    ],
)
def test_corrupt_baseline_is_rejected_before_benchmark_with_sanitized_error(
    path: tuple[str, ...],
    value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _load_benchmark()
    baseline = _load_json(BASELINE_PATH)
    _replace_nested(baseline, path, value)
    corrupt_path = tmp_path / "corrupt-baseline.json"
    corrupt_path.write_text(
        json.dumps(baseline, allow_nan=True),
        encoding="utf-8",
    )

    async def must_not_benchmark(**_kwargs):
        raise AssertionError("benchmark ran before baseline validation")

    monkeypatch.setattr(benchmark, "run_benchmark", must_not_benchmark)
    exit_code = benchmark.main(
        [
            "--mode",
            "qualification",
            "--samples",
            "1",
            "--warmups",
            "0",
            "--baseline",
            str(corrupt_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("benchmark error: invalid qualification baseline")
    assert "Traceback" not in captured.err
    assert str(corrupt_path) not in captured.err


@pytest.mark.parametrize(
    ("samples", "warmups", "valid"),
    [
        (MISSING, 5, False),
        (True, 5, False),
        ("30", 5, False),
        (29, 5, False),
        (30, MISSING, False),
        (30, False, False),
        (30, "5", False),
        (30, 4, False),
        (30, 5, True),
    ],
)
def test_qualification_rejects_inadequately_sampled_historical_baseline(
    samples: object,
    warmups: object,
    valid: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _load_benchmark()
    baseline = _load_json(BASELINE_PATH)
    _replace_nested(baseline, ("samples",), samples)
    _replace_nested(baseline, ("warmups",), warmups)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    benchmark_called = False

    async def record_benchmark(**_kwargs):
        nonlocal benchmark_called
        benchmark_called = True
        return {"budgets": {"overall_pass": True}}

    monkeypatch.setattr(benchmark, "run_benchmark", record_benchmark)
    exit_code = benchmark.main(
        [
            "--mode",
            "qualification",
            "--samples",
            "30",
            "--warmups",
            "5",
            "--baseline",
            str(baseline_path),
        ]
    )
    captured = capsys.readouterr()

    assert benchmark_called is valid
    assert exit_code == (0 if valid else 2)
    if valid:
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert captured.err.startswith(
            "benchmark error: invalid qualification baseline"
        )
        assert "Traceback" not in captured.err
        assert str(baseline_path) not in captured.err


@pytest.mark.asyncio
async def test_local_runner_is_machine_readable_and_never_opens_a_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()

    def reject_network(*_args, **_kwargs):
        raise AssertionError("local benchmark attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    result = await benchmark.run_benchmark(
        mode="baseline",
        samples=2,
        warmups=1,
        scratch_root=tmp_path,
    )

    assert set(result) == {
        "environment",
        "fixture_version",
        "samples",
        "warmups",
        "metrics",
        "budgets",
        "external_network",
    }
    assert result["samples"] == 2
    assert result["warmups"] == 1
    assert result["external_network"]["included_in_local_pass"] is False
    assert result["external_network"]["measured"] is False
    for metric in (
        result["metrics"]["first_token"]["control_ms"],
        result["metrics"]["first_token"]["candidate_ms"],
        result["metrics"]["finalization"]["standard_ms"],
        result["metrics"]["finalization"]["maximum_ms"],
        result["metrics"]["inspector_load"]["cold_ms"],
        result["metrics"]["inspector_load"]["warm_ms"],
        result["metrics"]["database_growth"]["bytes"],
        result["metrics"]["migration"]["messages_per_second"],
    ):
        assert set(metric) == {"median", "p95"}


@pytest.mark.asyncio
async def test_runner_consumes_each_representative_corpus_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    corpus = _load_json(CORPUS_PATH)
    monkeypatch.setattr(benchmark, "_load_fixture", lambda: corpus)

    result = await benchmark.run_benchmark(
        mode="baseline",
        samples=1,
        warmups=0,
        scratch_root=tmp_path,
    )

    assert "corpus_coverage" in result["metrics"]
    coverage = result["metrics"]["corpus_coverage"]
    assert set(coverage["generation"]["answer_cases"]) == {
        "unicode",
        "repeated_markers",
        "grouped_markers",
        "repaired_answer",
    }
    assert set(coverage["generation"]["source_kinds"]) == {
        "media",
        "note",
        "conversation",
    }
    assert coverage["generation"]["evidence_count"] == 1
    assert coverage["finalization"]["evidence_counts"] == [8, 64]
    assert coverage["inspector"]["evidence_count"] == 32
    assert set(coverage["inspector"]["storage_modes"]) == {
        "embedded",
        "server_reference",
        "ephemeral",
        "redacted",
    }
    assert set(coverage["migration"]["record_types"]) == {
        "EvidenceBundle",
        "CitationRef",
        "chat_rag_context_sidecar",
    }


@pytest.mark.asyncio
async def test_changing_selected_corpus_cases_changes_exercised_seam_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark()
    original = _load_json(CORPUS_PATH)
    changed = copy.deepcopy(original)
    changed["answers"][0]["body"] += " changed"
    changed["sources"][0]["text"] += " changed"
    changed["legacy_records"][0]["payload"]["query"] += " changed"

    monkeypatch.setattr(benchmark, "_load_fixture", lambda: original)
    original_result = await benchmark.run_benchmark(
        mode="baseline",
        samples=1,
        warmups=0,
        scratch_root=tmp_path / "original",
    )
    monkeypatch.setattr(benchmark, "_load_fixture", lambda: changed)
    changed_result = await benchmark.run_benchmark(
        mode="baseline",
        samples=1,
        warmups=0,
        scratch_root=tmp_path / "changed",
    )

    for family in ("first_token", "finalization", "migration"):
        assert "corpus_input_sha256" in original_result["metrics"][family]
        assert (
            original_result["metrics"][family]["corpus_input_sha256"]
            != changed_result["metrics"][family]["corpus_input_sha256"]
        )


def test_qualification_budget_cannot_pass_on_an_incompatible_environment() -> None:
    benchmark = _load_benchmark()
    metrics = benchmark.empty_passing_metrics()
    baseline = {
        "fixture_version": benchmark.FIXTURE_VERSION,
        "environment": {
            **benchmark.environment_metadata(),
            "provider": "different-provider",
        },
        "metrics": metrics,
    }

    budgets = benchmark.evaluate_budgets(
        metrics,
        baseline=baseline,
        mode="qualification",
    )

    assert budgets["environment_compatible"] is False
    assert budgets["overall_pass"] is False


def test_qualification_cannot_normalize_away_large_historical_regression() -> None:
    benchmark = _load_benchmark()
    environment = benchmark.environment_metadata()
    metrics = benchmark.empty_passing_metrics()
    metrics["first_token"]["control_ms"] = {"median": 100.0, "p95": 100.0}
    metrics["first_token"]["candidate_ms"] = {"median": 100.0, "p95": 100.0}
    baseline_metrics = benchmark.empty_passing_metrics()
    baseline_metrics["first_token"]["control_ms"] = {
        "median": 1.0,
        "p95": 1.0,
    }
    baseline_metrics["first_token"]["candidate_ms"] = {
        "median": 1.0,
        "p95": 1.0,
    }
    baseline = {
        "fixture_version": benchmark.FIXTURE_VERSION,
        "environment": environment,
        "metrics": baseline_metrics,
    }

    budgets = benchmark.evaluate_budgets(
        metrics,
        baseline=baseline,
        mode="qualification",
        environment=environment,
    )

    historical = budgets["checks"]["first_token"]["regressions"]["historical_v1"]
    assert historical["milliseconds"] == 99.0
    assert historical["percent"] == 9900.0
    assert budgets["checks"]["first_token"]["pass"] is False
    assert budgets["overall_pass"] is False


def test_qualification_passes_direct_compatible_historical_limits() -> None:
    benchmark = _load_benchmark()
    environment = benchmark.environment_metadata()
    metrics = benchmark.empty_passing_metrics()
    metrics["first_token"]["control_ms"] = {"median": 100.0, "p95": 100.0}
    metrics["first_token"]["candidate_ms"] = {"median": 110.0, "p95": 110.0}
    baseline_metrics = benchmark.empty_passing_metrics()
    baseline_metrics["first_token"]["control_ms"] = {
        "median": 100.0,
        "p95": 100.0,
    }
    baseline_metrics["first_token"]["candidate_ms"] = {
        "median": 100.0,
        "p95": 100.0,
    }
    baseline = {
        "fixture_version": benchmark.FIXTURE_VERSION,
        "environment": environment,
        "metrics": baseline_metrics,
    }

    budgets = benchmark.evaluate_budgets(
        metrics,
        baseline=baseline,
        mode="qualification",
        environment=environment,
    )

    assert budgets["checks"]["first_token"]["regressions"] == {
        "historical_v1": {"milliseconds": 10.0, "percent": 10.0},
        "in_process_control": {"milliseconds": 10.0, "percent": 10.0},
    }
    assert budgets["overall_pass"] is True


@pytest.mark.parametrize(
    ("samples", "warmups", "eligible", "reasons"),
    [
        (1, 0, False, ["insufficient_samples", "insufficient_warmups"]),
        (30, 5, True, []),
    ],
)
def test_qualification_eligibility_requires_thirty_samples_and_five_warmups(
    samples: int,
    warmups: int,
    eligible: bool,
    reasons: list[str],
) -> None:
    benchmark = _load_benchmark()
    environment = benchmark.environment_metadata()
    metrics = benchmark.empty_passing_metrics()
    baseline = {
        "fixture_version": benchmark.FIXTURE_VERSION,
        "environment": environment,
        "metrics": benchmark.empty_passing_metrics(),
    }

    budgets = benchmark.evaluate_budgets(
        metrics,
        baseline=baseline,
        mode="qualification",
        environment=environment,
        samples=samples,
        warmups=warmups,
    )

    assert budgets["qualification_eligible"] is eligible
    assert budgets["qualification_ineligibility_reasons"] == reasons
    assert all(check["pass"] for check in budgets["checks"].values())
    assert budgets["overall_pass"] is eligible


@pytest.mark.parametrize(
    ("changes", "failed_family"),
    [
        (
            [(("first_token", "candidate_ms", "p95"), 1.100001)],
            "first_token",
        ),
        (
            [
                (("first_token", "control_ms", "p95"), 1000.0),
                (("first_token", "candidate_ms", "p95"), 1025.001),
            ],
            "first_token",
        ),
        (
            [(("finalization", "standard_ms", "p95"), 75.001)],
            "finalization",
        ),
        (
            [(("finalization", "maximum_ms", "p95"), 250.001)],
            "finalization",
        ),
        (
            [(("inspector_load", "cold_ms", "p95"), 100.001)],
            "inspector_load",
        ),
        (
            [(("inspector_load", "warm_ms", "p95"), 25.001)],
            "inspector_load",
        ),
        (
            [(("trace_size", "aggregate_json_bytes"), (256 * 1024) + 1)],
            "trace_size",
        ),
        (
            [(("trace_size", "maximum_snapshot_bytes"), (64 * 1024) + 1)],
            "trace_size",
        ),
        (
            [(("trace_size", "governed_trace_bytes"), (4 * 1024 * 1024) + 1)],
            "trace_size",
        ),
        (
            [(("database_growth", "bytes", "p95"), 262145.351)],
            "database_growth",
        ),
        (
            [(("migration", "messages_per_second", "median"), 99.999)],
            "migration",
        ),
        (
            [
                (
                    (
                        "migration",
                        "duplicate_proxy_rows_after_restart",
                        "p95",
                    ),
                    0.001,
                )
            ],
            "migration",
        ),
    ],
)
def test_each_budget_sub_limit_rejects_a_just_over_candidate(
    changes: list[tuple[tuple[str, ...], float]],
    failed_family: str,
) -> None:
    benchmark = _load_benchmark()
    metrics = benchmark.empty_passing_metrics()
    for path, value in changes:
        _replace_nested(metrics, path, value)

    budgets = benchmark.evaluate_budgets(
        metrics,
        baseline=None,
        mode="baseline",
        samples=30,
        warmups=5,
    )

    assert budgets["checks"][failed_family]["pass"] is False
    assert budgets["overall_pass"] is False


def test_external_mode_requires_explicit_target_and_provider() -> None:
    benchmark = _load_benchmark()

    with pytest.raises(ValueError, match="target"):
        benchmark.validate_args(benchmark.parse_args(["--mode", "external"]))
    with pytest.raises(ValueError, match="provider"):
        benchmark.validate_args(
            benchmark.parse_args(
                [
                    "--mode",
                    "external",
                    "--external-target",
                    "https://example.invalid/health",
                ]
            )
        )


@pytest.mark.asyncio
async def test_external_measurement_uses_injected_resolver_and_is_not_a_local_gate() -> (
    None
):
    benchmark = _load_benchmark()
    calls: list[str] = []

    async def resolver(target: str, _timeout_seconds: float) -> None:
        calls.append(target)

    result = await benchmark.run_external_measurement(
        target="https://example.invalid/health",
        samples=2,
        warmups=1,
        timeout_seconds=1.0,
        resolver=resolver,
    )

    assert calls == ["https://example.invalid/health"] * 3
    assert result["external_network"]["measured"] is True
    assert result["external_network"]["included_in_local_pass"] is False
    assert set(result["external_network"]["latency_ms"]) == {"median", "p95"}
    assert result["environment"]["provider"] == benchmark.EXTERNAL_PROVIDER
    assert result["environment"]["network"] == "explicit-external-mode"
    assert "supported_envelope" not in result["environment"]
    assert result["budgets"]["overall_pass"] is None
    assert result["budgets"]["checks"] == {}
