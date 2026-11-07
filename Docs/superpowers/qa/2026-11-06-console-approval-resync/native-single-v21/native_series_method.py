"""Private fixed41-round method; pixels and owner facts are independent evidence."""

import hashlib
import math

LIMITS = {
    "rounds": 41,
    "rows_round": 64,
    "rows_process": 2624,
    "queue": 256,
    "row_bytes": 4096,
    "stage_bytes_process": 10747904,
    "captures_round": 64,
    "captures_process": 2624,
    "PNGs_round": 6,
    "failure_PNGs_round": 1,
    "PNG_bytes_round": 524288,
    "PNG_bytes_process": 22020096,
    "settlement_observations": 20,
    "settlement_seconds": 2,
    "control_spans_round": 32,
    "control_spans_setup": 16,
    "control_spans_process": 1328,
}


def runtime_round_limit(selection, authorization):
    diagnostic = selection.get("diagnostic_stack", False)
    if not isinstance(diagnostic, bool):
        raise RuntimeError("Explicit boolean diagnostic-stack flag required")
    mode = selection.get("method_mode", "validation")
    if mode == "qualification" and diagnostic:
        raise RuntimeError(
            "Diagnostic stacks are validation-only; qualification App held"
        )
    if mode == "validation":
        return 1
    if mode != "qualification" or type(authorization) is not dict:  # noqa: E721 - Exact boundary types reject bool/subclasses.
        raise RuntimeError(
            "Qualification runtime held: Root validation+freeze authorization missing"
        )
    if authorization.get("rounds") != 41 or any(
        authorization.get(key) is not True
        for key in (
            "qualified_method",
            "feedback_root_reviewed",
            "terminal_root_reviewed",
        )
    ):
        raise RuntimeError(
            "Qualification runtime held: exact Root method authorization required"
        )
    return 41


def pixels(cells):
    if (
        type(cells) not in (list, tuple)
        or len(cells) != 4
        or any(type(n) is not int for n in cells)  # noqa: E721 - Exact boundary types reject bool/subclasses.
        or min(cells[:2]) < 0
        or min(cells[2:]) <= 0
    ):
        raise ValueError("Invalid actual cell region")
    return [value * (8 if index % 2 == 0 else 16) for index, value in enumerate(cells)]


def patch(image, rect):
    if (
        image.mode != "RGBA"
        or type(rect) not in (list, tuple)
        or len(rect) != 4
        or any(type(n) is not int for n in rect)  # noqa: E721 - Exact boundary types reject bool/subclasses.
    ):
        raise ValueError("Invalid newRGBA patch")
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.width or y + h > image.height:
        raise ValueError("Partial actual patch")
    return hashlib.sha256(image.crop((x, y, x + w, y + h)).tobytes()).hexdigest()


def exact_text(image, rect, text, atlas):
    try:
        x, y, w, h = rect
        if image.mode != "RGBA" or h != 16 or w != 8 * len(text):
            return False
        patch(image, rect)
        return all(
            atlas.get(patch(image, [x + i * 8, y, 8, 16])) == char
            for i, char in enumerate(text)
        )
    except (KeyError, TypeError, ValueError):
        return False


def full_nonce(image, nonce, atlas):
    if (
        type(nonce) is not str  # noqa: E721 - Exact boundary types reject bool/subclasses.
        or len(nonce) != 32
        or any(char not in "0123456789abcdef" for char in nonce)
    ):
        return False
    matches = 0
    for y in range(0, image.height - 15, 16):
        line = "".join(
            atlas.get(patch(image, [x, y, 8, 16]), "?")
            for x in range(0, image.width - 7, 8)
        )
        matches += sum(
            line.startswith(nonce, index) for index in range(max(0, len(line) - 31))
        )
    return matches == 1


def healthy(state):
    from native_rehearsal_observation import check_scope

    check_scope(state)


def classify_feedback(image, state, expected, refs):
    answer = {
        "accepted_pixels": False,
        "source": "recomputed_new_capture_RGBA",
        "disabled_actions_valid": True,
    }
    try:
        healthy(state)
        view = state["snapshot"]
        nonce = expected["correlation"]
        if (
            view["current"]["correlation"] != nonce
            or view["card_identity"] != expected["card_identity"]
            or view.get("submitted") is not True
            or view.get("card_display") is not True
        ):
            return answer
        feedback = view["feedback"]
        if (
            len(feedback) != 1
            or feedback[0]["decision_state"] != "accepted"
            or feedback[0]["selected_scope"] != "approve_once"
        ):
            return answer
        rect = pixels(state["native"]["feedback_title_cell_region"])
        reference = refs["accepted_feedback"]
        actual = patch(image, rect)
        answer.update(
            title_RGBA_sha256=actual,
            title_pixel_region=rect,
            nonce_exact=full_nonce(image, nonce, refs["atlas"]),
        )
        answer["accepted_pixels"] = (
            answer["nonce_exact"]
            and rect[2:] == reference["pixel_size"]
            and actual == reference["RGBA_sha256"]
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        answer["refusal"] = "actual_scope_identity_or_pixels"
    return answer


def classify_terminal(image, state, expected, refs):
    answer = {
        "settled_pixels": False,
        "candidate_pixels": False,
        "generation_exact": False,
        "source": "recomputed_new_capture_RGBA",
    }
    try:
        healthy(state)
        view = state["snapshot"]
        facts = state["settlement"]
        correlation = expected["correlation"]
        if any(
            view["current"].get(key) != expected[key]
            for key in ("correlation", "run_id", "session_id")
        ):
            return answer
        required = (
            "owner_terminal",
            "session_matches",
            "card_retired",
            "header_terminal",
            "send_terminal",
            "stop_hidden",
            "run_segment_absent",
            "title_generation_matches",
        )
        if view["current"]["correlation"] != correlation or any(
            facts.get(key) is not True for key in required
        ):
            return answer
        rois = facts["rois"]
        if set(rois) != {"generation", "header", "send", "provider"}:
            return answer
        answer["generation_exact"] = exact_text(
            image, pixels(rois["generation"]), correlation[:12], refs["terminal_atlas"]
        )
        if not answer["generation_exact"]:
            return answer
        patches = {
            name: patch(image, pixels(rois[name]))
            for name in ("header", "send", "provider")
        }
        sizes = {name: pixels(rois[name])[2:] for name in patches}
        answer.update(candidate_pixels=True, patches=patches, patch_pixel_sizes=sizes)
        reference = refs.get("terminal_patches")
        answer["settled_pixels"] = type(reference) is dict and all(  # noqa: E721 - Exact boundary types reject bool/subclasses.
            reference[name]["RGBA_sha256"] == patches[name]
            and reference[name]["pixel_size"] == sizes[name]
            for name in patches
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        answer["refusal"] = "actual_scope_identity_or_pixels"
    return answer


def settlement_pair(first, second):
    return bool(
        first
        and second
        and first.get("generation_exact") is True
        and second.get("generation_exact") is True
        and first.get("patches")
        and first["patches"] == second.get("patches")
        and type(first.get("qpc_end")) is int  # noqa: E721 - Exact boundary types reject bool/subclasses.
        and type(second.get("qpc_start")) is int  # noqa: E721 - Exact boundary types reject bool/subclasses.
        and first["qpc_end"] < second["qpc_start"] < second.get("qpc_end", 0)
    )


def planned_outcomes(records, planned):
    if planned not in (1, 41):
        raise ValueError("Fixed validation1 or qualification41 only")
    by_index = {}
    for row in records:
        index = row["round_index"]
        if type(index) is not int or not 0 <= index < planned or index in by_index:  # noqa: E721 - Exact boundary types reject bool/subclasses.
            raise ValueError("Duplicate/out-of-range actual round")
        by_index[index] = row
    return [
        by_index.get(
            index,
            {
                "round_index": index,
                "status": "missing",
                "reason": "Not run; no replacement or dropped denominator",
            },
        )
        for index in range(planned)
    ]


def input_components(rows, commit_ns, feedback_ns):
    down = next(
        (row["timestamp_ns"] for row in rows if row.get("kind") == "MouseDown"), None
    )
    up = next(
        (row["timestamp_ns"] for row in rows if row.get("kind") == "MouseUp"), None
    )
    if (
        any(type(n) is not int for n in (down, up, commit_ns, feedback_ns))  # noqa: E721 - Exact boundary types reject bool/subclasses.
        or not down <= up <= commit_ns <= feedback_ns
    ):
        return None
    return {
        "earliest_app_input_to_feedback_ns": feedback_ns - down,
        "MouseDown_to_MouseUp_ns": up - down,
        "committing_MouseUp_to_handler_ns": commit_ns - up,
        "MouseUp_to_feedback_ns": feedback_ns - up,
    }


def nearest_rank_p95(values):
    if not values:
        return None
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError("Finite actual samples required")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def qpc_perf_interval(qpc_value, clocks):
    lower = []
    upper = []
    frequency = clocks[0]["qpc_frequency"]
    if any(clock["qpc_frequency"] != frequency for clock in clocks):
        raise ValueError("QPC frequency changed")
    for clock in clocks:
        for before, perf, after in clock["brackets"]:
            if before > after:
                raise ValueError("Invalid QPC/perf bracket")
            lower.append(perf - (after * 1000000000 + frequency - 1) // frequency)
            upper.append(perf - before * 1000000000 // frequency)
    if not lower:
        raise ValueError("Clock calibration missing")
    return [
        qpc_value * 1000000000 // frequency + min(lower),
        (qpc_value * 1000000000 + frequency - 1) // frequency + max(upper),
    ]


def authorized_round_limit(selection, run):
    # Explicit QA qualification receipt is independent of product authority.
    # Validation stays hard1 even if the caller supplies41 in another field.
    authorization = selection.get("method_authorization")
    limit = runtime_round_limit(selection, authorization)
    if limit == 1:
        return limit
    import json
    from pathlib import Path

    source = Path(selection["source_root"]).resolve()
    freeze_path = Path(selection["freeze_path"])
    frozen_raw = freeze_path.read_bytes()
    frozen = json.loads(frozen_raw)
    if (
        Path(authorization.get("source_root", "")).resolve() != source
        or authorization.get("revision") != selection.get("revision")
        or authorization.get("freeze_sha256") != hashlib.sha256(frozen_raw).hexdigest()
    ):
        raise RuntimeError("Qualification source/freeze binding missing")
    if (
        not isinstance(frozen, dict)
        or frozen.get("qualification_authorized") is not True
        or Path(frozen.get("source_root", "")).resolve() != source
        or frozen.get("revision") != selection.get("revision")
    ):
        raise RuntimeError(
            "Actual freeze does not explicitly authorize qualification for selected source/revision"
        )
    expected = selection["helper_sha256"]
    if authorization.get("runtime_sha256") != expected:
        raise RuntimeError("Qualification runtime byte binding missing")
    for name, digest in expected.items():
        if (
            not (Path(run) / name).resolve().is_relative_to(Path(run).resolve())
            or hashlib.sha256((Path(run) / name).read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError("Qualification runtime bytes changed")
    for key in ("feedback_review", "terminal_review"):
        reference = authorization.get(key)
        if type(reference) is not dict:  # noqa: E721 - Exact boundary types reject bool/subclasses.
            raise RuntimeError("Actual Root live reference review missing")
        path = Path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise RuntimeError("Root reference review changed")
        review = json.loads(path.read_text(encoding="utf-8"))
        if review.get("observed") is not True or not review.get("reviewer"):
            raise RuntimeError("Root live reference classification missing")
        png = Path(review["source_png"])
        if hashlib.sha256(png.read_bytes()).hexdigest() != review["png_sha256"]:
            raise RuntimeError("Actual Root reference PNG changed")
    terminal = json.loads(
        Path(authorization["terminal_review"]["path"]).read_text(encoding="utf-8")
    )
    if type(terminal.get("patches")) is not dict or set(terminal["patches"]) != {  # noqa: E721 - Exact boundary types reject bool/subclasses.
        "header",
        "send",
        "provider",
    }:
        raise RuntimeError("Root settled ROI reference missing")
    return limit


def bind_diagnostic_stack(selection, passed):
    """Bind explicit child argv to the recorded QA-only diagnostic flag."""
    if (
        "diagnostic_stack" not in selection
        or not isinstance(selection["diagnostic_stack"], bool)
        or passed is not selection["diagnostic_stack"]
    ):
        raise RuntimeError("Missing/mismatched explicit selected/child diagnostic flag")
    return passed
