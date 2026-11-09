"""Offline exact-frame classifications; never launch App/browser or infer pixels."""

import argparse, base64, copy, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from browser_presented_frames import presented_frames
from browser_series_method import (
    eligible,
    select_observed_frame,
    terminal_pair,
    sample_metrics,
    summarize,
    validate_clock_series,
    validate_capture_series,
)


def read_json(path, limit):
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("Offline_metadata_byte_bound")
    return json.loads(raw)


def actual_image(path, raw):
    if len(raw) > 1024 * 1024:
        raise ValueError("Actual_JPEG_1MiB_cap")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Actual_frame_image_changed")
    else:
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def decode_frames(directory, path, calibration, prefix):
    data = read_json(path, 64 * 1024 * 1024)
    frames, diagnostics = presented_frames(
        data["traceEvents"], page_pid=calibration["page_pid"]
    )
    output = []
    for index, row in enumerate(frames):
        row = dict(row)
        raw = base64.b64decode(row.pop("snapshot"), validate=True)
        image = directory / f"{prefix}-frame-{index}.jpg"
        row.update(image=image.name, image_sha256=actual_image(image, raw))
        output.append(row)
    return output, diagnostics


def earliest_observed(directory, frames, reviews, correlation, state):
    candidates = sorted(
        (f for f in frames if eligible(f)), key=lambda f: f["presented_us"]
    )
    if not candidates:
        raise ValueError("No_actual_presented_candidate")
    for frame in candidates:
        ack = [
            r
            for r in reviews
            if r.get("image") == frame["image"]
            and r.get("image_sha256") == frame["image_sha256"]
        ]
        if len(ack) != 1:
            raise ValueError("Earlier_eligible_frame_not_independently_classified")
        review = ack[0]
        if (
            review.get("observed") is not True
            or not review.get("reviewer")
            or review.get("correlation") != correlation
        ):
            raise ValueError("Stale_or_unobserved_frame_classification")
        if review.get("state") == "other":
            if review.get("negative_classified") is not True:
                raise ValueError("Earlier_frame_negative_classification_missing")
            continue
        if review.get("state") != state:
            raise ValueError("Unexpected_frame_classification")
        if state == "actionable" and review.get("enabled_action_verified") is not True:
            raise ValueError("Enabled_action_not_observed")
        if state == "feedback" and (
            review.get("truthful_current_generation_verified") is not True
            or review.get("visible_feedback_state")
            not in ("applying", "received", "accepted", "running", "completed")
        ):
            raise ValueError("Truthful_feedback_state_not_observed")
        return select_observed_frame(
            directory, frames, review, correlation=correlation, state=state
        )
    raise ValueError("No_matching_independently_observed_frame")


def collect(run):
    selection = read_json(run / "source-selection.json", 1024 * 1024)
    for name, digest in selection["fixture_hashes"].items():
        path = (run / name).resolve()
        if (
            not path.is_relative_to(run)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Runtime_code_changed_after_capture")
    result = read_json(run / "smoke-result.json", 16 * 1024 * 1024)
    planned = result["planned_rounds"]
    if (
        result.get("control_span_overflow")
        or result.get("control_span_observer_failure")
        or any(
            r.get("control_span_overflow")
            for r in result.get("round_control_spans", [])
        )
    ):
        raise ValueError("Actual_control_observer_loss_or_overflow")
    if planned not in (1, 41):
        raise ValueError("Only_fixed1_or41")
    clocks = result.get("capture_clock_segments", [])
    paths = [(run / p).resolve() for p in result.get("trace_paths", [])]
    if any(not p.is_relative_to(run) for p in paths):
        raise ValueError("Trace_path_not_owned")
    clock_report = validate_clock_series(clocks)
    capture_report = validate_capture_series(paths)
    records = []
    prefixes = set()
    requests = []
    for original in result["runs"]:
        row = copy.deepcopy(original)
        index = row["round_index"]
        directory = run / f"round-{index:03d}"
        correlation = row["correlation"]
        if correlation[:12] in prefixes:
            raise ValueError("Actual_nonce_prefix_collision")
        prefixes.add(correlation[:12])
        if row.get("status") not in (
            "complete",
            "captured_pending_offline_classification",
        ):
            records.append(row)
            continue
        card = directory / row["actionable_trace"]
        feedback = directory / row["feedback_trace"]
        if (
            hashlib.sha256(card.read_bytes()).hexdigest()
            != row["actionable_trace_sha256"]
            or hashlib.sha256(feedback.read_bytes()).hexdigest()
            != row["feedback_trace_sha256"]
        ):
            raise ValueError("Raw_trace_changed")
        card_frames, card_diag = decode_frames(
            directory, card, row["clock_calibration"], "approve-card"
        )
        feedback_frames, feedback_diag = decode_frames(
            directory, feedback, row["feedback_clock_calibration"], "approve-feedback"
        )
        request = {
            "round_index": index,
            "correlation": correlation,
            "qualified": False,
            "product_samples": 0,
            "actionable_frames": card_frames,
            "feedback_terminal_frames": feedback_frames,
            "semantic_actionable": row["semantic_actionable"],
            "terminal_states": row["terminal_states"],
            "ready_server_ns": row["ready_server_ns"],
            "instructions": "Classify EVERY eligible frame preceding the earliest matching actionable/feedback. Each actualimage/hash gets observed/reviewer/correlation. Matching actionable: state=actionable,full_nonce_verified,enabled_action_verified. Matching feedback: state=feedback,full_nonce_verified,truthful_current_generation_verified,visible_feedback_state enum applying/received/accepted/running/completed. Earlier presented frames that do not match: state=other,negative_classified=true. terminal=[two distinct ordered eligible Ready/disabledSend/hiddenStop/noRun/currentQA12 frames with state=terminal,visible_nonce_prefix,terminal_UI_verified]. Never infer contents from semantic metadata or repeat image hashes.",
        }
        request_path = directory / "offline-frame-review-required.json"
        if not request_path.exists():
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        else:
            if read_json(request_path, 2 * 1024 * 1024) != request:
                raise ValueError("Offline_review_request_changed")
        requests.append(str(request_path))
        review_path = directory / "offline-frame-review.json"
        if not review_path.exists():
            row["status"] = "missing_offline_classification"
            records.append(row)
            continue
        reviews = read_json(review_path, 256 * 1024)
        try:
            if reviews.get("correlation") != correlation:
                raise ValueError("Stale_offline_review")
            actionable = earliest_observed(
                directory, card_frames, reviews["actionable"], correlation, "actionable"
            )
            first_feedback = earliest_observed(
                directory, feedback_frames, reviews["feedback"], correlation, "feedback"
            )
            terminal = terminal_pair(
                directory,
                feedback_frames,
                reviews["terminal"],
                correlation=correlation,
                ready_server_ns=row["ready_server_ns"],
                calibration=row["feedback_clock_calibration"],
            )
            if any(
                r.get("stage") == "app_input_prequeue"
                and r.get("kind") not in ("MouseDown", "MouseUp")
                for r in row["owner_stages"]
            ):
                raise ValueError("Unexpected_actual_input_traffic_separate_condition")
            row.update(
                status="complete",
                metrics=sample_metrics(
                    row["owner_stages"],
                    correlation,
                    actionable,
                    first_feedback,
                    row["clock_calibration"],
                    row["feedback_clock_calibration"],
                    row["browser_input"][0],
                ),
                selected_presented_frame=actionable,
                selected_feedback=first_feedback,
                terminal_presented=terminal,
                offline_review_sha256=hashlib.sha256(
                    review_path.read_bytes()
                ).hexdigest(),
            )
        except (ValueError, RuntimeError, KeyError) as error:
            row.update(
                status="invalid_offline_classification",
                classification_error_type=type(error).__name__,
            )
        records.append(row)
    report = summarize(records, planned)
    report.update(
        clock_series=clock_report,
        capture_series=capture_report,
        review_requests=requests,
        raw_runtime_completed=result.get("completed"),
        full_source_worker_effect_process_closure_pending=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.output.exists():
        raise RuntimeError("Preserve_existing_offline_report_use_unique_output")
    report = collect(run)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "qualified": False,
                "product_samples": 0,
                "review_requests": report["review_requests"],
                "output": str(args.output),
            }
        )
    )
