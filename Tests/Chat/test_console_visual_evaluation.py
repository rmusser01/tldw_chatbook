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
SUPPORT_MATRIX_PATH = CORPUS_PATH.with_name("support-matrix.json")
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
        response_text: str | None = None,
        synthetic_fallback: bool = False,
    ) -> None:
        self.corpus = corpus
        self.valid_response = valid_response
        self.include_usage = include_usage
        self.response_text = response_text
        self.synthetic_fallback = synthetic_fallback
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
        if self.synthetic_fallback and signals is not None:
            signals.mark_synthetic_fallback()
        if self.response_text is not None:
            yield self.response_text
        elif self.valid_response:
            yield _response(self.corpus)
        else:
            yield "not valid evaluator JSON"
        if signals is not None:
            signals.close_usage_call()


def _resolution(
    *,
    provider: str = "openai",
    model: str = "gpt-5.6-terra",
    base_url: str = "https://api.openai.com/v1",
    execution_key: str | None = None,
) -> ConsoleProviderResolution:
    return ConsoleProviderResolution(
        provider=provider,
        base_url=base_url,
        model=model,
        ready=True,
        execution_key=execution_key or provider,
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
        parse_failure_reason=None if quality is not None else "invalid_json",
        response_sha256="a" * 64,
        ocr_fidelity=None,
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
        output_enforcement="provider_json_schema",
        evaluation_mode="context_use",
    )


def _legacy_matrix_payload() -> dict[str, Any]:
    """Return strict evaluator-v1 evidence without depending on live evidence."""

    data = json.loads(
        build_visual_support_matrix((_report("legacy-model", ready=True),)).to_json()
    )
    data["schema_version"] = 1
    data["generated_by"] = "chatbook-visual-evaluator-v1"
    data["eligible_models"] = []
    report = data["reports"][0]
    report["schema_version"] = 1
    report["evaluator_version"] = "chatbook-visual-evaluator-v1"
    report["default_enablement_ready"] = False
    report["recommendation"] = "not_recommended"
    report.pop("output_enforcement")
    report.pop("evaluation_mode")
    for representation in (report["text"], report["visual"]):
        representation.pop("parse_failure_reason")
        representation["ocr_fidelity"] = 1.0
    report["visual"].update(
        {
            "parse_status": "invalid",
            "ocr_fidelity": None,
            "code_math_recovery": None,
            "instruction_recall": None,
            "adversarial_text_safe": None,
        }
    )
    return data


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
    assert all(request.response_format is not None for request in gateway.requests)
    assert report.model == "gpt-5.6-terra"
    assert report.output_enforcement == "provider_json_schema"
    response_format = gateway.requests[0].response_format
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "answers",
        "adversarial_instruction_followed",
    }
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["answers"]["additionalProperties"] is False
    assert (
        gateway.requests[0].messages_payload[-1]
        == (gateway.requests[1].messages_payload[-1])
    )
    active_request = str(gateway.requests[0].messages_payload[-1]["content"])
    assert "answer every downstream probe" in active_request
    assert "complete historical transcript body" not in active_request
    assert report.text.input_tokens == 1_000
    assert report.visual.input_tokens == 400
    assert report.measured_usage_complete is True
    assert report.token_reduction_ratio == pytest.approx(0.6)
    assert report.evaluation_mode == "context_use"
    assert report.visual.ocr_fidelity is None
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
    assert report.visual.parse_failure_reason == "invalid_json"
    assert report.visual.ocr_fidelity is None
    assert report.visual.code_math_recovery is None
    assert report.visual.instruction_recall is None
    assert report.visual.adversarial_text_safe is None
    assert report.default_enablement_ready is False
    assert report.recommendation == "not_recommended"


@pytest.mark.parametrize(
    ("response_text", "expected_reason"),
    [
        ("", "empty_response"),
        ("not json", "invalid_json"),
        ("[]", "unexpected_top_level_shape"),
        (
            '{"transcript_text":"extracted history","answers":{},'
            '"adversarial_instruction_followed":false}',
            "unexpected_top_level_shape",
        ),
        (
            '{"answers":[],' '"adversarial_instruction_followed":false}',
            "invalid_answers_shape",
        ),
        (
            '{"answers":{},' '"adversarial_instruction_followed":false}',
            "probe_id_mismatch",
        ),
    ],
)
def test_invalid_responses_persist_only_stable_content_free_reason(
    response_text: str,
    expected_reason: str,
) -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    gateway = _ImmediateGateway(corpus, response_text=response_text)

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

    assert report.text.parse_failure_reason == expected_reason
    assert report.visual.parse_failure_reason == expected_reason
    assert report.default_enablement_ready is False
    persisted = report.to_json()
    if response_text:
        assert response_text not in persisted
    assert corpus.source_text not in persisted


def test_response_shape_failures_are_classified_without_response_content() -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    valid = json.loads(_response(corpus))
    extra_field = dict(valid, unexpected="private raw value")
    invalid_answer = json.loads(_response(corpus))
    invalid_answer["answers"][corpus.probes[0].probe_id] = 7
    invalid_flag = json.loads(_response(corpus))
    invalid_flag["adversarial_instruction_followed"] = "false"

    for payload, expected_reason in (
        (extra_field, "unexpected_top_level_shape"),
        (invalid_answer, "invalid_answer_value"),
        (invalid_flag, "invalid_adversarial_flag"),
    ):
        raw_response = json.dumps(payload, ensure_ascii=False)
        gateway = _ImmediateGateway(corpus, response_text=raw_response)
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

        assert report.visual.parse_failure_reason == expected_reason
        assert raw_response not in report.to_json()
        assert report.default_enablement_ready is False


@pytest.mark.parametrize(
    "resolution",
    [
        _resolution(provider="anthropic", model="claude-sonnet-4"),
        _resolution(base_url="https://proxy.example/v1"),
        _resolution(model="gpt-5.6-sol"),
        _resolution(model="gpt-5.6-terra-preview"),
        _resolution(model="gpt-4o-2024-05-13"),
        _resolution(model="gpt-4o-audio-preview"),
    ],
)
def test_unsupported_routes_remain_honestly_prompt_only(
    resolution: ConsoleProviderResolution,
) -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    gateway = _ImmediateGateway(corpus)

    report = _run_immediate(
        evaluate_visual_compaction_model(
            gateway=gateway,
            resolution=resolution,
            corpus=corpus,
            evaluated_at_utc="2026-08-11T00:00:00+00:00",
            vision_available=True,
            max_images=10,
        )
    )

    assert report.output_enforcement == "prompt_only"
    assert all(request.response_format is None for request in gateway.requests)


def test_synthetic_fallback_has_content_free_reason_and_never_passes() -> None:
    corpus = load_visual_evaluation_corpus(CORPUS_PATH)
    gateway = _ImmediateGateway(corpus, synthetic_fallback=True)

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

    assert report.visual.parse_status == "synthetic_fallback"
    assert report.visual.parse_failure_reason == "synthetic_fallback"
    assert report.visual.ocr_fidelity is None
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


def test_evaluator_v1_matrix_remains_strictly_loadable(tmp_path: Path) -> None:
    original = _legacy_matrix_payload()
    path = tmp_path / "legacy-matrix.json"
    path.write_text(json.dumps(original), encoding="utf-8")

    matrix = load_visual_support_matrix(path)

    assert matrix.schema_version == 1
    assert matrix.reports[0].schema_version == 1
    assert matrix.reports[0].output_enforcement == "prompt_only"
    assert matrix.reports[0].evaluation_mode == "transcription_recovery"
    assert matrix.reports[0].visual.parse_failure_reason == "legacy_unspecified"
    assert json.loads(matrix.to_json()) == original


def test_checked_in_evaluator_v2_matrix_remains_strictly_loadable() -> None:
    original = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))

    matrix = load_visual_support_matrix(SUPPORT_MATRIX_PATH)

    assert matrix.schema_version == 2
    assert len(matrix.reports) == 1
    assert matrix.reports[0].model == "gpt-5.6-terra"
    assert matrix.reports[0].evaluation_mode == "transcription_recovery"
    assert matrix.reports[0].output_enforcement == "provider_json_schema"
    assert matrix.reports[0].measured_usage_complete is True
    assert matrix.reports[0].default_enablement_ready is False
    assert json.loads(matrix.to_json()) == original


def test_new_matrix_preserves_legacy_reports_without_making_them_eligible(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy-matrix.json"
    legacy_path.write_text(json.dumps(_legacy_matrix_payload()), encoding="utf-8")
    legacy = load_visual_support_matrix(legacy_path).reports[0]
    current = _report("new-model", ready=True)
    matrix = build_visual_support_matrix((legacy, current))
    path = tmp_path / "mixed-matrix.json"
    path.write_text(matrix.to_json(), encoding="utf-8")

    loaded = load_visual_support_matrix(path)

    assert loaded == matrix
    assert loaded.schema_version == 3
    assert {report.schema_version for report in loaded.reports} == {1, 3}
    assert loaded.eligible_models == ("openai/new-model",)


def test_ready_evaluator_v2_report_cannot_make_v3_matrix_eligible(
    tmp_path: Path,
) -> None:
    current = _report("legacy-ready", ready=True)
    payload = json.loads(build_visual_support_matrix((current,)).to_json())
    payload["schema_version"] = 2
    payload["generated_by"] = "chatbook-visual-evaluator-v2"
    report = payload["reports"][0]
    report["schema_version"] = 2
    report["evaluator_version"] = "chatbook-visual-evaluator-v2"
    report.pop("evaluation_mode")
    for representation in (report["text"], report["visual"]):
        representation["ocr_fidelity"] = 1.0
    legacy_path = tmp_path / "legacy-v2-matrix.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = load_visual_support_matrix(legacy_path).reports[0]

    assert legacy.default_enablement_ready is True
    assert legacy.evaluation_mode == "transcription_recovery"
    matrix = build_visual_support_matrix((legacy,))
    assert matrix.schema_version == 3
    assert matrix.eligible_models == ()


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
        replace(
            passing.visual,
            parse_status="invalid",
            parse_failure_reason="invalid_json",
        )

    no_savings_visual = replace(passing.visual, input_tokens=1_100)
    no_savings = replace(
        passing,
        visual=no_savings_visual,
        token_reduction_ratio=-0.1,
        default_enablement_ready=False,
        recommendation="not_recommended",
    )
    assert no_savings.default_enablement_ready is False

    with pytest.raises(ValueError, match="newer report schemas"):
        replace(build_visual_support_matrix((passing,)), schema_version=2)


def test_cli_validates_confirmation_and_output_before_loading_config(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        confirm_billable=False,
        max_output_tokens=4_096,
        output=tmp_path / "matrix.json",
        replace=False,
        provider="openai",
        model="gpt-5.6-terra",
    )
    with pytest.raises(ValueError, match="exactly two"):
        validate_live_request(args)

    args.confirm_billable = True
    args.max_output_tokens = 0
    with pytest.raises(ValueError, match="between 1 and 16384"):
        validate_live_request(args)

    args.max_output_tokens = 4_096
    args.output.write_text(
        build_visual_support_matrix((_report("gpt-5.6-terra", ready=True),)).to_json(),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="--replace"):
        validate_live_request(args)

    args.model = "gpt-4.1"
    assert validate_live_request(args) == args.output.resolve()

    args.model = "gpt-5.6-terra"
    args.replace = True
    assert validate_live_request(args) == args.output.resolve()


@pytest.mark.parametrize(
    ("arguments", "expected_returncode"),
    [
        (["--help"], 0),
        (["--provider", "openai", "--model", "gpt-5.6-terra"], 2),
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
