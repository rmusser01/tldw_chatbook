"""Approval receipts must refuse incomplete or misleading measurements."""

import pytest

from Tests.Benchmarks import console_approval_latency as measurement


def samples(count=40, **changes):
    return [
        dict(
            transport="native",
            clock_qualified=True,
            first_use=False,
            feedback_ms=float(value),
            actionable_ms=100.0,
            **changes,
        )
        for value in range(1, count + 1)
    ]


def test_nearest_rank_p95_uses_complete_sample_count():
    report = measurement.summarize_approval_timings(samples())
    assert report["sample_count"] == 40
    assert report["feedback_p95_ms"] == 38.0
    assert report["feedback_median_ms"] == 20.5
    assert report["feedback_max_ms"] == 40.0
    assert report["feedback_budget_ms"] == 100.0
    assert report["actionable_budget_ms"] == 200.0
    assert report["qualified"] is True


def test_missing_paint_cannot_qualify():
    values = samples()
    values[0]["feedback_ms"] = None
    report = measurement.summarize_approval_timings(values)
    assert report["sample_count"] == 40
    assert report["qualified"] is False
    assert report["missing_boundaries"]["feedback_ms"] == 1
    assert report["feedback_p95_ms"] is None


def test_uncalibrated_clock_pair_is_unverified():
    values = samples()
    values[0]["clock_qualified"] = False
    assert measurement.summarize_approval_timings(values)["qualified"] is False


def test_disallowed_content_is_rejected():
    recorder = measurement.ApprovalTimingRecorder()
    correlation = recorder.new_correlation()
    recorder.record(
        "input_arrived",
        correlation=correlation,
        clock="python_perf_counter",
        timestamp_ns=1,
    )
    with pytest.raises(ValueError):
        recorder.record(
            "arbitrary argument content",
            correlation=correlation,
            clock="python_perf_counter",
            timestamp_ns=2,
        )
    with pytest.raises(ValueError):
        recorder.record(
            "input_arrived",
            correlation="external-id",
            clock="python_perf_counter",
            timestamp_ns=2,
        )
    with pytest.raises(TypeError):
        recorder.record(
            "input_arrived",
            correlation=correlation,
            clock="python_perf_counter",
            timestamp_ns=2,
            arguments="must not be accepted",
        )


def test_compositor_only_receipt_cannot_claim_native_transport():
    values = samples()
    for value in values:
        value["transport"] = "compositor"
    assert measurement.summarize_approval_timings(values)["qualified"] is False


def test_first_use_is_separate_and_does_not_fill_warm_minimum():
    values = samples(39)
    values.append(
        dict(
            transport="native",
            clock_qualified=True,
            first_use=True,
            feedback_ms=999.0,
            actionable_ms=999.0,
        )
    )
    report = measurement.summarize_approval_timings(values)
    assert report["sample_count"] == 39
    assert report["qualified"] is False
    assert report["feedback_max_ms"] == 39.0
    assert report["first_use_sample_count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("transport", "external-content"),
        ("feedback_ms", float("nan")),
        ("actionable_ms", -1),
        ("first_use", "false"),
        ("clock_qualified", 1),
    ],
)
def test_invalid_sample_values_are_rejected(field, value):
    values = samples()
    values[0][field] = value
    with pytest.raises(ValueError):
        measurement.summarize_approval_timings(values)


def test_sample_content_and_mixed_transports_cannot_qualify():
    values = samples()
    values[0]["arguments"] = "unsupported content"
    with pytest.raises(ValueError):
        measurement.summarize_approval_timings(values)
    values[0].pop("arguments")
    values[0]["transport"] = "browser"
    assert measurement.summarize_approval_timings(values)["qualified"] is False


def test_recorder_clock_timestamp_and_snapshot_are_guarded():
    recorder = measurement.ApprovalTimingRecorder()
    correlation = recorder.new_correlation()
    for clock, timestamp in [
        ("external-content", 1),
        ("python_perf_counter", -1),
        ("python_perf_counter", True),
    ]:
        with pytest.raises(ValueError):
            recorder.record(
                "input_arrived",
                correlation=correlation,
                clock=clock,
                timestamp_ns=timestamp,
            )
    recorder.record(
        "input_arrived",
        correlation=correlation,
        clock="python_perf_counter",
        timestamp_ns=123,
    )
    snapshot = recorder.snapshot()
    assert snapshot == [
        dict(
            stage="input_arrived",
            correlation=correlation,
            clock="python_perf_counter",
            timestamp_ns=123,
        )
    ]
    snapshot[0]["stage"] = "external-content"
    assert recorder.snapshot()[0]["stage"] == "input_arrived"


def test_recorder_is_bounded_and_rejects_overflow():
    recorder = measurement.ApprovalTimingRecorder()
    correlation = recorder.new_correlation()
    for value in range(4096):
        recorder.record(
            "input_arrived",
            correlation=correlation,
            clock="python_perf_counter",
            timestamp_ns=value,
        )
    with pytest.raises(ValueError):
        recorder.record(
            "input_arrived",
            correlation=correlation,
            clock="python_perf_counter",
            timestamp_ns=4096,
        )
    assert len(recorder.snapshot()) == 4096


def test_qualification_is_distinct_from_budget_success():
    values = samples()
    for value in values:
        value["feedback_ms"] = 101.0
        value["actionable_ms"] = 201.0
    report = measurement.summarize_approval_timings(values)
    assert report["qualified"] is True
    assert report["feedback_p95_ms"] == 101.0
    assert report["actionable_p95_ms"] == 201.0


def test_empty_receipt_has_no_distribution():
    report = measurement.summarize_approval_timings([])
    assert report["sample_count"] == 0
    assert report["qualified"] is False
    assert report["feedback_p95_ms"] is None
    assert report["actionable_median_ms"] is None
