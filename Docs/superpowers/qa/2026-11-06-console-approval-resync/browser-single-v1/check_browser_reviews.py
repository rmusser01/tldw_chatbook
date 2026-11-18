"""Pure metadata logic controls, NOT real image/pixel/Root observations."""

import copy, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect_browser_series import earliest_observed
from browser_series_method import select_observed_frame, sample_metrics, summarize


def control_output():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if not output.is_relative_to(source / ".superpowers"):
        raise RuntimeError(
            "Control_outputs_must_stay_inside_selectedsource_superpowers"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def refused(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid_metadata_control_released")


p = control_output() / "review-controls"
p.mkdir(exist_ok=True)
corr = "a" * 32
frames = []
for index in range(3):
    path = p / f"NOT-A-SCREENSHOT-{index}.bin"
    path.write_bytes(f"pure metadata fixture {index}".encode())
    frames.append(
        {
            "image": path.name,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_id": 1,
            "frame_sequence": index,
            "identity_matches": 1,
            "eligible_matches": 1,
            "presented_us": 1000 * (index + 1),
            "rejection": None,
        }
    )
frames[0].update(
    identity_matches=0, eligible_matches=0, presented_us=None, rejection="missing"
)
other = {
    "observed": True,
    "reviewer": "unit-fixture-not-Root",
    "correlation": corr,
    "image": frames[1]["image"],
    "image_sha256": frames[1]["image_sha256"],
    "state": "other",
    "negative_classified": True,
}
match = {
    "observed": True,
    "reviewer": "unit-fixture-not-Root",
    "correlation": corr,
    "image": frames[2]["image"],
    "image_sha256": frames[2]["image_sha256"],
    "state": "feedback",
    "full_nonce_verified": True,
    "truthful_current_generation_verified": True,
    "visible_feedback_state": "accepted",
}
assert earliest_observed(p, frames, [other, match], corr, "feedback") == frames[2]
refused(lambda: earliest_observed(p, frames, [match], corr, "feedback"))
stale = copy.deepcopy(match)
stale["correlation"] = "b" * 32
refused(lambda: earliest_observed(p, frames, [other, stale], corr, "feedback"))
refused(lambda: earliest_observed(p, frames, [other, match, match], corr, "feedback"))
untrusted = copy.deepcopy(match)
untrusted["full_nonce_verified"] = False
refused(lambda: earliest_observed(p, frames, [other, untrusted], corr, "feedback"))
rejected = copy.deepcopy(match)
rejected.update(image=frames[0]["image"], image_sha256=frames[0]["image_sha256"])
refused(
    lambda: select_observed_frame(
        p, frames, rejected, correlation=corr, state="feedback"
    )
)
refused(
    lambda: select_observed_frame(
        p, frames + [frames[2]], match, correlation=corr, state="feedback"
    )
)
original = (p / frames[2]["image"]).read_bytes()
(p / frames[2]["image"]).write_bytes(b"changed")
refused(
    lambda: select_observed_frame(p, frames, match, correlation=corr, state="feedback")
)
(p / frames[2]["image"]).write_bytes(original)
clock = {
    "envelope": {
        "trace_page": {"lower_ms": 99, "upper_ms": 101},
        "server_page": {"lower_ms": 99, "upper_ms": 100},
    }
}
card = copy.deepcopy(frames[2])
card["presented_us"] = 200000
feedback = copy.deepcopy(frames[2])
feedback["presented_us"] = 450000
stamps = {
    "request_complete": 100000000,
    "local_checks_complete": 110000000,
    "card_commit_enter": 331000000,
    "resolver_entered": 332000000,
    "worker_released": 333000000,
}
rows = [
    {"stage": key, "timestamp_ns": value, "correlation": corr}
    for key, value in stamps.items()
]
rows += [
    {
        "stage": "app_input_prequeue",
        "kind": kind,
        "timestamp_ns": value,
        "correlation": corr,
    }
    for kind, value in [("MouseDown", 130000000), ("MouseUp", 330000000)]
]
metrics = sample_metrics(
    rows,
    corr,
    card,
    feedback,
    clock,
    clock,
    {"timestamp_ms": 29, "handler_ms": 29.1, "trusted": True},
)
assert metrics["local_feedback"]["upper_ms"] == 321
assert (
    metrics["app_MouseDown_to_MouseUp_ms"] == 200
    and metrics["committing_MouseUp_to_handler_ms"] == 1
)
assert metrics["transport_inclusive_feedback"]["upper_ms"] == 322
assert (
    metrics["prior_presented_output_to_card"] is None
    and metrics["real_model_continuation"] is None
)
refused(
    lambda: sample_metrics(
        rows + rows[:1],
        corr,
        card,
        feedback,
        clock,
        clock,
        {"timestamp_ms": 29, "trusted": True},
    )
)
refused(
    lambda: sample_metrics(
        rows, corr, card, feedback, clock, clock, {"timestamp_ms": 29, "trusted": False}
    )
)
records = []
for index in range(41):
    m = {
        key: {"upper_ms": index}
        for key in (
            "local_feedback",
            "local_actionable",
            "transport_inclusive_feedback",
        )
    }
    records.append({"round_index": index, "status": "complete", "metrics": m})
summary = summarize(records, 41)
d = summary["distributions"]["local_feedback"]
assert (
    d["sample_count"] == 40
    and d["median_ms"] == 20.5
    and d["p95_ms"] == 38
    and d["maximum_ms"] == 40
)
assert len(summary["first_use"]) == 1 and summary["qualified"] is False
missing = summarize(records[:-1], 41)["distributions"]["local_feedback"]
assert missing["missing_warm"] == 1 and missing["p95_ms"] is None
print(
    json.dumps(
        {
            "status": "PASS",
            "pure_logic_only_no_real_images_or_Root_reviews": True,
            "earliest_review_coverage": True,
            "missing_stale_duplicate_unpresented_false_nonce_changed_bytes_refused": True,
            "earliest_Down_preserved_200ms_delivery_gap": True,
            "browser_transport_separate": True,
            "firstuse_and_all40_denominator": True,
            "app_browser_model_socket_launches": 0,
            "qualified": False,
            "product_samples": 0,
        }
    )
)
