"""Finite actual-RGBA/method controls. No App/native launch or product imports."""

import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--source-root", required=True, type=Path)
_parser.add_argument("--output-root", required=True, type=Path)
_parser.add_argument("--dependencies", type=Path)
_args = _parser.parse_args()
OUTPUT = _args.output_root.resolve()
if not OUTPUT.is_relative_to(_args.source_root.resolve() / ".superpowers"):
    raise RuntimeError(
        "Pure control outputs must stay inside selectedsource/.superpowers"
    )
OUTPUT.mkdir(parents=True, exist_ok=True)
if _args.dependencies is not None:
    sys.path.insert(0, str(_args.dependencies.resolve()))
from PIL import Image  # noqa: E402 - Selected-source/profile/dependency guard runs first.
from native_rehearsal_observation import classify_pixels  # noqa: E402 - Selected-source/profile/dependency guard runs first.

HERE = Path(__file__).resolve().parent
refs = json.loads((HERE / "exact-pixel-refs.json").read_text(encoding="utf-8"))
source = HERE / "native_series_method.py"
module = None
if source.exists():
    spec = importlib.util.spec_from_file_location("native_series_method", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
results = {}
error = None
try:
    fixture = json.loads(
        (HERE / "reference-fixtures/accepted-feedback-state.json").read_text(
            encoding="utf-8"
        )
    )
    png = HERE / fixture["export_png"]
    image = Image.open(png).convert("RGBA")
    state = fixture["state"]
    expected = {
        "correlation": state["snapshot"]["current"]["correlation"],
        "card_identity": state["snapshot"]["card_identity"],
    }
    if module is None:
        old = classify_pixels(image, expected["correlation"], [0, 0, 1, 1], refs)
        result = {"accepted_pixels": old["nonce_exact"] and old["action_enabled"]}
    else:
        result = module.classify_feedback(image, state, expected, refs)
    assert result[
        "accepted_pixels"
    ], "Actual Root009 accepted feedback must not require enabled permission actions"
    results["actual_Root009_disabled_accepted_feedback_passes"] = True
    for key, value in [
        ("correlation", "0" * 32),
        ("card_identity", ["other-round", 1]),
    ]:
        wrong = dict(expected)
        wrong[key] = value
        assert not module.classify_feedback(image, state, wrong, refs)[
            "accepted_pixels"
        ]
    results["wrong_nonce_and_round_refuse"] = True
    changed = image.copy()
    x, y, w, h = module.pixels(state["native"]["feedback_title_cell_region"])
    changed.putpixel((x, y), (1, 2, 3, 255))
    assert not module.classify_feedback(changed, state, expected, refs)[
        "accepted_pixels"
    ]
    results["one_changed_actual_RGBA_pixel_refuses"] = True
    wrong = dict(refs)
    wrong["caller_patch_hash"] = refs["accepted_feedback"]["RGBA_sha256"]
    blank = Image.new("RGBA", image.size)
    assert not module.classify_feedback(blank, state, expected, wrong)[
        "accepted_pixels"
    ]
    results["caller_hash_cannot_replace_new_pixel_proof"] = True
    assert module.runtime_round_limit({"method_mode": "validation"}, None) == 1
    try:
        module.runtime_round_limit({"method_mode": "qualification"}, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "41rounds admitted without actual Root qualification authorization"
        )
    auth = {
        "qualified_method": True,
        "rounds": 41,
        "feedback_root_reviewed": True,
        "terminal_root_reviewed": True,
    }
    assert module.runtime_round_limit({"method_mode": "qualification"}, auth) == 41
    for field in auth:
        bad = dict(auth)
        bad[field] = None
        try:
            module.runtime_round_limit({"method_mode": "qualification"}, bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Incomplete authorization accepted:" + field)
    results["validation_hard1_qualification_requires_exact_Root_flags"] = True
    assert (
        module.LIMITS["rows_process"] == 2624
        and module.LIMITS["stage_bytes_process"] == 10747904
    )
    assert (
        module.LIMITS["captures_process"] == 2624
        and module.LIMITS["PNG_bytes_process"] == 22020096
    )
    assert (
        module.LIMITS["rows_round"] == 64 and module.LIMITS["PNG_bytes_round"] == 524288
    )
    result = module.planned_outcomes(
        [
            {"round_index": 0, "status": "valid"},
            {"round_index": 1, "status": "invalid"},
        ],
        41,
    )
    assert len(result) == 41 and [row["status"] for row in result[:3]] == [
        "valid",
        "invalid",
        "missing",
    ]
    assert len([row for row in result[1:] if row["status"] == "missing"]) == 39
    results["finite_caps_and_fixed_denominator_no_replacement"] = True
    assert module.settlement_pair(
        {"qpc_start": 1, "qpc_end": 2, "patches": {"a": "x"}, "generation_exact": True},
        {"qpc_start": 3, "qpc_end": 4, "patches": {"a": "x"}, "generation_exact": True},
    )
    for second in (
        {"qpc_start": 1, "qpc_end": 2, "patches": {"a": "x"}, "generation_exact": True},
        {"qpc_start": 3, "qpc_end": 4, "patches": {"a": "y"}, "generation_exact": True},
        {
            "qpc_start": 3,
            "qpc_end": 4,
            "patches": {"a": "x"},
            "generation_exact": False,
        },
    ):
        assert not module.settlement_pair(
            {
                "qpc_start": 1,
                "qpc_end": 2,
                "patches": {"a": "x"},
                "generation_exact": True,
            },
            second,
        )
    results["two_new_generation_bound_matching_frames_required"] = True
    samples = [10, 20, 30, 40]
    assert module.nearest_rank_p95(samples) == 40
    assert module.nearest_rank_p95([]) is None
    down = 100
    up = 280
    commit = 291
    feedback = 350
    values = module.input_components(
        [
            {"kind": "MouseDown", "timestamp_ns": down},
            {"kind": "MouseUp", "timestamp_ns": up},
        ],
        commit,
        feedback,
    )
    assert values == {
        "earliest_app_input_to_feedback_ns": 250,
        "MouseDown_to_MouseUp_ns": 180,
        "committing_MouseUp_to_handler_ns": 11,
        "MouseUp_to_feedback_ns": 70,
    }
    results["earliest_Down_primary_and_separate_Up_queue_interval"] = True
    assert len(refs["terminal_atlas"]) == 16 and set(
        refs["terminal_atlas"].values()
    ) == set("0123456789abcdef")
    terminal_image = Image.open(HERE / "reference-fixtures/header-current.png").convert(
        "RGBA"
    )
    assert module.exact_text(
        terminal_image, [144, 0, 96, 16], "f65a5053cfd8", refs["terminal_atlas"]
    )
    assert not module.exact_text(
        terminal_image, [144, 0, 96, 16], "0" * 12, refs["terminal_atlas"]
    )
    assert refs["terminal_patches"] is None
    assert not module.classify_terminal(
        terminal_image, {"settlement": {}}, "f65a5053cfd88a033f734cf5f1cb40d8", refs
    )["settled_pixels"]
    results[
        "actual_header_palette_full16_and_transitional026_not_terminal_reference"
    ] = True
    # Synthetic owner/reference contract only, NOT a claim that Root026 is terminal.
    expectation = {
        "correlation": "f65a5053cfd88a033f734cf5f1cb40d8",
        "run_id": "controlled-run",
        "session_id": "controlled-session",
    }
    controlled = {
        "snapshot": {
            "current": dict(expectation),
            "theme": "textual-dark",
            "size": [120, 40],
            "inspect_open": False,
        },
        "native": {"font": [8, 16]},
        "observer_health": {
            "loss": False,
            "write_failure": False,
            "observer_failure": False,
        },
        "settlement": {
            key: True
            for key in (
                "owner_terminal",
                "session_matches",
                "card_retired",
                "header_terminal",
                "send_terminal",
                "stop_hidden",
                "run_segment_absent",
                "title_generation_matches",
            )
        },
    }
    controlled["settlement"]["rois"] = {
        "generation": [18, 0, 12, 1],
        "header": [0, 0, 1, 1],
        "send": [1, 0, 1, 1],
        "provider": [2, 0, 1, 1],
    }
    testrefs = dict(refs)
    testrefs["terminal_patches"] = {
        name: {
            "RGBA_sha256": module.patch(terminal_image, module.pixels(region)),
            "pixel_size": module.pixels(region)[2:],
        }
        for name, region in controlled["settlement"]["rois"].items()
        if name != "generation"
    }
    assert module.classify_terminal(terminal_image, controlled, expectation, testrefs)[
        "settled_pixels"
    ]
    for key in ("run_id", "session_id", "correlation"):
        wrong = dict(expectation)
        wrong[key] = "wrong"
        assert not module.classify_terminal(
            terminal_image, controlled, wrong, testrefs
        )["settled_pixels"]
    for key in ("owner_terminal", "card_retired", "stop_hidden", "run_segment_absent"):
        controlled["settlement"][key] = False
        assert not module.classify_terminal(
            terminal_image, controlled, expectation, testrefs
        )["settled_pixels"]
        controlled["settlement"][key] = True
    results[
        "synthetic_terminal_contract_refuses_wrongrun_session_generation_and_active_UI"
    ] = True
    selection = {
        "method_mode": "qualification",
        "method_authorization": dict(auth, source_root="wrong"),
        "source_root": str(HERE),
        "freeze_path": str(HERE / "exact-pixel-refs.json"),
        "revision": "controlled",
    }
    try:
        module.authorized_round_limit(selection, HERE)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unbound source admitted41")
    results["qualification_requires_frozen_source_and_runtime_byte_binding"] = True
except BaseException as caught:
    error = {"type": type(caught).__name__, "reason": str(caught)}
facts = {
    "status": "GREEN" if error is None else "RED",
    "qualified": False,
    "product_samples": 0,
    "results": results,
    "error": error,
    "scope": "Actual retained RGBA + finite pure metadata/clock controls only; no product imports, App, Win32, GUI, external/service/model traffic or performance pass.",
}
(OUTPUT / "series-method-control-results.json").write_text(
    json.dumps(facts, indent=2), encoding="utf-8"
)
print(json.dumps(facts, indent=2))
assert error is None, error
assert not any(
    name.startswith(("tldw_chatbook", "tldw_profile_core", "textual", "ctypes"))
    for name in sys.modules
)
