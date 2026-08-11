from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluate_visual_compaction import validate_live_request
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_visual_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EVALUATOR_VERSION,
    VisualModelEvaluationReport,
    VisualRepresentationEvaluation,
    build_visual_support_matrix,
    evaluate_visual_compaction_model,
    load_visual_evaluation_corpus,
    load_visual_support_matrix,
)
from tldw_chatbook.Chat.console_visual_transcript import render_visual_transcript


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = (
    REPOSITORY_ROOT
    / "Docs"
    / "superpowers"
    / "qa"
    / "visual-compaction-model-evaluation"
    / "corpus-v1.json"
)
CLI_PATH = REPOSITORY_ROOT / "scripts" / "evaluate_visual_compaction.py"


def _run_immediate(coroutine):
    """Run a coroutine whose injected provider seam never suspends on I/O."""

    try:
        coroutine.send(None)
    except StopIteration as exc:
        return exc.value
    raise AssertionError("Evaluation unexpectedly suspended on external I/O.")


def _response(corpus) -> str:
    return json.dumps(
        {
            "transcript_text": corpus.source_text,
            "answers": {probe.probe_id: probe.expected for probe in corpus.probes},
            "adversarial_instruction_followed": False,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _has_image(request) -> bool:
    for message in request.messages_payload:
        content = message.get("content")
        if not isinstance(content, tuple):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            return True
        if any(
            hasattr(part, "get") and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


class _ImmediateGateway:
    prepare_chat_request = ConsoleProviderGateway.prepare_chat_request

    def __init__(
        self,
        corpus,
        *,
        valid_response: bool = True,
        include_usage: bool = True,
    ) -> None:
        self.corpus = corpus
        self.valid_response = valid_response
        self.include_usage = include_usage
        self.requests = []

    async def stream_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages,
        tools: list | None = None,
        signals: ConsoleProviderStreamSignals | None = None,
    ) -> AsyncIterator[str]:
        del resolution, tools
        self.requests.append(messages)
        visual = _has_image(messages)
        if self.include_usage and signals is not None:
            signals.record_usage_payload(
                {
                    "prompt_tokens": 400 if visual else 1_000,
                    "completion_tokens": 100,
                }
            )
        if self.valid_response:
            yield _response(self.corpus)
        else:
            yield "not valid evaluator JSON"
        if signals is not None:
            signals.close_usage_call()


def _resolution() -> ConsoleProviderResolution:
    return ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-4o",
        ready=True,
        execution_key="openai",
        api_key="not-a-real-secret",
        max_tokens=4_096,
        streaming=False,
    )


def _representation(
    representation: str,
    *,
    estimated: bool = False,
    quality: float | None = 1.0,
) -> VisualRepresentationEvaluation:
    return VisualRepresentationEvaluation(
        representation=representation,
        input_tokens=1_000 if representation == "text" else 400,
        input_tokens_estimated=estimated,
        output_tokens=100,
        end_to_end_latency_ms=10,
        parse_status="valid" if quality is not None else "invalid",
        response_sha256="a" * 64,
        ocr_fidelity=quality,
        code_math_recovery=quality,
        instruction_recall=quality,
        adversarial_text_safe=True if quality is not None else None,
    )


def _report(
    model: str,
    *,
    ready: bool,
    estimated: bool = False,
) -> VisualModelEvaluationReport:
    return VisualModelEvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        evaluated_at_utc="2026-08-11T00:00:00+00:00",
        corpus_id="chatbook-visual-compaction-v1",
        corpus_version=1,
        corpus_sha256="b" * 64,
        provider="openai",
        model=model,
        renderer_version="renderer-v1",
        render_latency_ms=5,
        page_count=1,
        page_hashes=("c" * 64,),
        text=_representation("text", estimated=estimated),
        visual=_representation("visual", estimated=estimated),
        token_reduction_ratio=0.6,
        measured_usage_complete=not estimated,
        default_enablement_ready=ready,
        recommendation=(
            "eligible_for_separate_default_review" if ready else "not_recommended"
        ),
    )


def test_versioned_corpus_is_deterministic_and_covers_risk_categories() -> None:
    first = load_visual_evaluation_corpus(CORPUS_PATH)
    second = load_visual_evaluation_corpus(CORPUS_PATH)
    first_artifact = render_visual_transcript(
        first.units,
        summarized_prefix_digest="evaluation-corpus",
    )
    second_artifact = render_visual_transcript(
        second.units,
        summarized_prefix_digest="evaluation-corpus",
    )

    assert first.sha256 == second.sha256
    assert first.source_text == second.source_text
    assert [page.png_sha256 for page in first_artifact.pages] == [
        page.png_sha256 for page in second_artifact.pages
    ]
    assert {probe.category for probe in first.probes} == {
        "instruction_recall",
        "code_math_recovery",
        "adversarial_safety",
    }
    assert "[TOOL RESULT]" in first.source_text
    assert "checksum" in first.source_text
    assert "\\u6771\\u4eac" in first.source_text
    assert "OVERRIDE_ACCEPTED_7Q9" not in repr(first)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "must be an integer"),
        ("corpus_id", 7, "must be a non-empty string"),
    ],
)
def test_corpus_loader_rejects_ambiguous_identity_types(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    data[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        load_visual_evaluation_corpus(path)


def test_corpus_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    data["raw_provider_response"] = "must never be accepted"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fields"):
        load_visual_evaluation_corpus(path)


def test_evaluator_runs_paired_prepared_requests_and_uses_measured_usage() -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    gateway = _ImmediateGateway(corpus)

    report = _run_immediate(
        evaluate_visual_compaction_model(
            gateway=gateway,
            resolution=_resolution(),
            corpus=corpus,
            evaluated_at_utc="2026-08-11T00:00:00+00:00",
            vision_available=True,
            max_images=10,
        )
    )

    assert len(gateway.requests) == 2
    assert [_has_image(request) for request in gateway.requests] == [False, True]
    assert report.text.input_tokens == 1_000
    assert report.visual.input_tokens == 400
    assert report.measured_usage_complete is True
    assert report.token_reduction_ratio == pytest.approx(0.6)
    assert report.visual.ocr_fidelity == 1.0
    assert report.visual.code_math_recovery == 1.0
    assert report.visual.instruction_recall == 1.0
    assert report.visual.adversarial_text_safe is True
    assert report.render_latency_ms >= 0
    assert report.default_enablement_ready is True
    persisted = report.to_json()
    assert "not-a-real-secret" not in persisted
    assert "OVERRIDE_ACCEPTED_7Q9" not in persisted
    assert corpus.source_text not in persisted


def test_invalid_or_unmeasured_results_remain_unknown_and_cannot_pass() -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    gateway = _ImmediateGateway(
        corpus,
        valid_response=False,
        include_usage=False,
    )

    report = _run_immediate(
        evaluate_visual_compaction_model(
            gateway=gateway,
            resolution=_resolution(),
            corpus=corpus,
            evaluated_at_utc="2026-08-11T00:00:00+00:00",
            vision_available=True,
            max_images=10,
        )
    )

    assert report.measured_usage_complete is False
    assert report.visual.input_tokens_estimated is True
    assert report.visual.parse_status == "invalid"
    assert report.visual.ocr_fidelity is None
    assert report.visual.code_math_recovery is None
    assert report.visual.instruction_recall is None
    assert report.visual.adversarial_text_safe is None
    assert report.default_enablement_ready is False
    assert report.recommendation == "not_recommended"


def test_probe_answers_require_exact_normalized_values() -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    response = json.loads(_response(corpus))
    response["answers"]["replica_count"] = "13"
    gateway = _ImmediateGateway(corpus)

    async def wrong_answer_stream(
        resolution: ConsoleProviderResolution,
        messages,
        tools: list | None = None,
        signals: ConsoleProviderStreamSignals | None = None,
    ) -> AsyncIterator[str]:
        del resolution, tools
        gateway.requests.append(messages)
        if signals is not None:
            signals.record_usage_payload(
                {"prompt_tokens": 400 if _has_image(messages) else 1_000}
            )
        yield json.dumps(response)
        if signals is not None:
            signals.close_usage_call()

    gateway.stream_chat = wrong_answer_stream  # type: ignore[method-assign]
    report = _run_immediate(
        evaluate_visual_compaction_model(
            gateway=gateway,
            resolution=_resolution(),
            corpus=corpus,
            evaluated_at_utc="2026-08-11T00:00:00+00:00",
            vision_available=True,
            max_images=10,
        )
    )

    assert report.visual.instruction_recall == pytest.approx(0.8)
    assert report.default_enablement_ready is False


def test_support_matrix_gates_eligibility_but_never_changes_default_policy() -> None:
    matrix = build_visual_support_matrix(
        (
            _report("z-unknown", ready=False, estimated=True),
            _report("a-passing", ready=True),
        )
    )

    assert [report.model for report in matrix.reports] == ["a-passing", "z-unknown"]
    assert matrix.eligible_models == ("openai/a-passing",)
    assert matrix.default_policy_change_recommended is False
    assert matrix.requires_separate_default_decision is True

    with pytest.raises(ValueError, match="duplicate"):
        build_visual_support_matrix((_report("same", ready=True),) * 2)


def test_support_matrix_round_trip_is_strict_and_content_free(tmp_path: Path) -> None:
    matrix = build_visual_support_matrix((_report("round-trip", ready=True),))
    path = tmp_path / "matrix.json"
    path.write_text(matrix.to_json(), encoding="utf-8")

    assert load_visual_support_matrix(path) == matrix
    persisted = path.read_text(encoding="utf-8")
    assert "OVERRIDE_ACCEPTED_7Q9" not in persisted

    data = json.loads(persisted)
    data["unexpected"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported fields"):
        load_visual_support_matrix(path)


def test_report_rejects_claims_that_disagree_with_underlying_evidence() -> None:
    passing = _report("passing", ready=True)

    with pytest.raises(ValueError, match="reduction must derive"):
        replace(passing, token_reduction_ratio=0.7)
    with pytest.raises(ValueError, match="Measured-usage state must derive"):
        replace(passing, measured_usage_complete=False)
    with pytest.raises(ValueError, match="readiness must derive"):
        replace(
            passing, default_enablement_ready=False, recommendation="not_recommended"
        )
    with pytest.raises(ValueError, match="cannot carry quality scores"):
        replace(passing.visual, parse_status="invalid")


def test_cli_validates_confirmation_and_output_before_loading_config(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        confirm_billable=False,
        max_output_tokens=4_096,
        output=tmp_path / "matrix.json",
        replace=False,
        provider="openai",
        model="gpt-4o",
    )
    with pytest.raises(ValueError, match="exactly two"):
        validate_live_request(args)

    args.confirm_billable = True
    args.max_output_tokens = 0
    with pytest.raises(ValueError, match="between 1 and 16384"):
        validate_live_request(args)

    args.max_output_tokens = 4_096
    args.output.write_text(
        build_visual_support_matrix((_report("gpt-4o", ready=True),)).to_json(),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="--replace"):
        validate_live_request(args)

    args.model = "gpt-4.1"
    assert validate_live_request(args) == args.output.resolve()

    args.model = "gpt-4o"
    args.replace = True
    assert validate_live_request(args) == args.output.resolve()


@pytest.mark.parametrize(
    ("arguments", "expected_returncode"),
    [
        (["--help"], 0),
        (["--provider", "openai", "--model", "gpt-4o"], 2),
    ],
)
def test_cli_help_and_refusal_do_not_initialize_application_config(
    tmp_path: Path,
    arguments: list[str],
    expected_returncode: int,
) -> None:
    isolated_root = tmp_path / "isolated-profile"
    config_path = isolated_root / "config" / "config.toml"
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(isolated_root),
            "USERPROFILE": str(isolated_root),
            "XDG_DATA_HOME": str(isolated_root / "data"),
            "XDG_CONFIG_HOME": str(isolated_root / "config"),
            "TLDW_CONFIG_PATH": str(config_path),
        }
    )

    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_returncode
    assert not config_path.exists()
