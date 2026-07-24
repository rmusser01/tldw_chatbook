"""Contracts for the network-free RAG citation provenance benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    REPO_ROOT / "Helper_Scripts" / "Benchmarks" / "rag_citation_provenance_benchmark.py"
)
FIXTURE_ROOT = REPO_ROOT / "Tests" / "fixtures" / "rag_citation_provenance"
MANIFEST_PATH = FIXTURE_ROOT / "manifest_v1.json"
CORPUS_PATH = FIXTURE_ROOT / "corpus_v1.json"


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
        measured = {
            case["expected"]: benchmark.materialized_boundary_size(case)
            for case in cases
        }
        assert measured == {"accept": limit, "reject": limit + 1}


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


def test_qualification_normalizes_historical_ttfb_to_its_current_control() -> None:
    benchmark = _load_benchmark()
    environment = benchmark.environment_metadata()
    metrics = benchmark.empty_passing_metrics()
    metrics["first_token"]["control_ms"] = {"median": 2.0, "p95": 2.0}
    metrics["first_token"]["candidate_ms"] = {"median": 2.0, "p95": 2.0}
    baseline_metrics = benchmark.empty_passing_metrics()
    baseline_metrics["first_token"]["control_ms"] = {
        "median": 0.5,
        "p95": 0.5,
    }
    baseline_metrics["first_token"]["candidate_ms"] = {
        "median": 0.5,
        "p95": 0.5,
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

    assert budgets["checks"]["first_token"]["regressions"]["historical_v1"] == {
        "milliseconds": 0.0,
        "percent": 0.0,
    }
    assert budgets["overall_pass"] is True
