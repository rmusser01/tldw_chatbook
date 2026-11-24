"""Grade numerical Windows adapter evidence offline. No audio dependencies."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

RATE = 48000
TAIL = 0.020
SEEK = 0.250
STOP = 1.5
CASES = {
    "natural_500ms": (24000, 0),
    "natural_120ms": (5760, 0),
    "natural_2500ms": (120000, 0),
    "resume_1230ms": (120000, 59040),
    "near_eof_140ms": (120000, 113280),
    "near_eof_60ms": (120000, 117120),
    "near_eof_one_frame": (120000, 119999),
    "exact_eof": (120000, 120000),
    "startup_stop": (120000, 0),
    "active_stop": (120000, 0),
    "padding_stop": (24000, 0),
    "resume_saved_bookmark": (120000, None),
    "early_zero_exit": (120000, 0),
}
PAIRS = (
    ("controlled", "controlled", "active"),
    ("controlled", "controlled", "startup"),
    ("controlled", "controlled", "padding"),
    ("controlled", "pcm", "active"),
    ("pcm", "controlled", "active"),
    ("pcm", "pcm", "active"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def num(obj, key):
    value = obj[key]
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {key}")
    return value


def confirmed(row, *keys):
    for key in keys:
        require(row[key] is True, key)


def quiet(window):
    require(window["status"] == "pass", "unqualified quiet window")
    if window["packet_frames"] == 0:
        require(
            window["evidence"] == "monitored_no_packets" and window["rms"] is None,
            "unmonitored silence",
        )
    else:
        require(
            window["evidence"] == "monitored_packets"
            and window["packet_frames"] > 0
            and 0 <= num(window, "rms") <= 0.0002,
            "channel noise",
        )


def capture(row, baseline):
    confirmed(row, "capture_reaped")
    require(row["capture_error"] is None, "capture error")
    quiet(row[baseline])
    cfg, metrics = row["capture_configuration"], row["capture_metrics"]
    require(
        0 < num(cfg, "device_period_s")
        and 0 <= num(cfg, "qpc_calibration_bracket_s") <= 0.001,
        "capture calibration",
    )
    bound = min(0.15, num(cfg, "actual_buffer_frames") / RATE)
    require(0 < num(metrics, "poll_gap_max_s") <= bound, "capture polling")
    require(
        metrics["timestamp_error_count"] == metrics["discontinuity_count"] == 0
        and metrics["discontinuity_raw_count"]
        == metrics["session_origin_discontinuity_count"]
        in (0, 1),
        "capture continuity",
    )


def match(row, name, diag_name, start, end=None):
    m, d = row[name], row[diag_name]
    require(
        d["stage"] == "matched" and d["timeline_stage"] == "contiguous",
        "unmatched waveform",
    )
    require(0 <= num(d, "packet_clock_error_max_s") <= 0.005, "packet clock")

    # Reconstruct the two independently located windows before trusting the exported summary.
    def frame(key):
        value = d[key]
        require(type(value) is int and value >= 0, "invalid " + key)
        return value

    begin, finish = frame("active_start_frame"), frame("active_end_frame")
    early, late = frame("early_capture_frame"), frame("tail_capture_frame")
    first, last = frame("early_source_frame"), frame("tail_source_frame")
    size, samples = frame("window_frames"), frame("capture_frames")
    count = row["capture_metrics"]["packet_count"]
    source_frames = row.get("content_frames", 120000 if name == "old_match" else 24000)
    margin = round(RATE * 0.008)
    require(
        type(count) is int
        and count > 0
        and frame("content_frames") == samples
        and samples <= count * num(row["capture_configuration"], "actual_buffer_frames")
        and RATE * 0.030 <= frame("active_frames") <= finish - begin <= samples
        and begin <= early < early + size <= late < late + size <= finish <= samples
        and early == begin + margin
        and late == finish - margin - size
        and size == min(round(RATE * 0.045), (finish - begin - 2 * margin) // 2)
        and size >= RATE * 0.012
        and first <= last
        and last + size <= source_frames,
        "invalid matched packet or frame geometry",
    )
    drift = ((last - first) - (late - early)) / RATE
    expected = {
        "start_source_s": (first - early + begin) / RATE,
        "end_source_s": (last + finish - late) / RATE,
        "capture_onset_s": begin / RATE,
        "capture_end_s": finish / RATE,
        "alignment_drift_s": drift,
        "early_score": num(d, "early_score"),
        "tail_score": num(d, "tail_score"),
    }
    require(
        all(abs(num(m, k) - v) <= 1e-9 for k, v in expected.items())
        and abs(num(d, "alignment_drift_s") - drift) <= 1e-9,
        "matcher summary disagrees with frame evidence",
    )
    for fields in (d, m):
        require(
            min(num(fields, "early_score"), num(fields, "tail_score")) >= 0.70
            and abs(num(fields, "alignment_drift_s")) <= TAIL,
            "correlation or drift",
        )
    onset, tail = num(m, "start_source_s"), num(m, "end_source_s")
    require(
        0 < num(m, "capture_onset_qpc_s") < num(m, "capture_end_qpc_s")
        and onset < tail,
        "waveform order",
    )
    length = num(m, "capture_end_s") - num(m, "capture_onset_s")
    require(
        abs(length - (tail - onset)) <= TAIL
        and abs(length - (m["capture_end_qpc_s"] - m["capture_onset_qpc_s"])) < 1e-7,
        "inconsistent source or capture edges",
    )
    require(abs(onset - start) <= SEEK, "source start")
    if end is not None:
        require(abs(tail - end) <= TAIL, "missing source tail")
    if name == "waveform_match":
        e = row["completion"]["evidence"]
        end_key = (
            "retired_at_s" if row["completion"]["kind"] == "stopped" else "reaped_at_s"
        )
        check_lifetime(m, num(e, "launched_at_s"), num(e, end_key))
    return m


def check_lifetime(m, launch, end):
    require(
        launch - TAIL
        <= m["capture_onset_qpc_s"]
        < m["capture_end_qpc_s"]
        <= end + TAIL,
        "waveform outside output lifetime",
    )


def terminal(completion, kind, reason, generation, count):
    require(
        (completion["kind"], completion["reason"], completion["generation"])
        == (kind, reason, generation),
        "wrong terminal or generation",
    )
    bookmark, e = completion["bookmark"], completion["evidence"]
    frame = bookmark["frame"]
    require(
        re.fullmatch("[0-9a-f]{64}", bookmark["sha256"])
        and bookmark["sample_rate"] == RATE
        and type(frame) is int
        and 0 <= frame <= count,
        "invalid bookmark",
    )
    if kind in ("natural", "already_at_end"):
        require(frame == count, "logical EOF")
    retired = num(e, "retired_at_s")
    if kind == "already_at_end":
        require(
            e["launched_at_s"] is e["reaped_at_s"] is e["stop_at_s"] is None
            and retired > 0,
            "EOF launched",
        )
        return
    launched, reaped = num(e, "launched_at_s"), num(e, "reaped_at_s")
    require(0 < launched < reaped <= retired, "stale retirement")
    if kind == "stopped":
        stop = num(e, "stop_at_s")
        require(
            launched <= stop <= reaped and retired - stop <= STOP, "stop observer bound"
        )
    else:
        require(e["stop_at_s"] is None, "unexpected Stop")
    if kind == "natural":
        require(
            count <= num(e, "last_physical_frame") <= count + 12000,
            "missing terminal progress",
        )


def fixture(facts, count, final_mode, digest=None):
    require(
        facts["native_mode"] == final_mode == "0o600"
        and facts["sample_rate"] == RATE
        and facts["logical_frames"] == facts["source_frames"] == count
        and facts["physical_frames"] == count + 12000
        and facts["padding_frames"] == 12000
        and re.fullmatch("[0-9a-f]{64}", facts["sha256"]),
        "fixture posture or frames",
    )
    if digest is not None:
        require(facts["sha256"] == digest, "fixture digest")


def grade_control(row, stopped=None, *, promoted=False):
    require(not row.get("error") and not row.get("stop_trigger_missed"), "run failed")
    case = row["case"]
    count, start = CASES[case]
    if case == "resume_saved_bookmark":
        require(
            stopped is not None and stopped["case"] == "active_stop",
            "missing source Stop",
        )
        grade_control(stopped, promoted=promoted)
        bookmark = stopped["completion"]["bookmark"]
        start = bookmark["frame"]
        require(
            row["source_stop_case"] == "active_stop"
            and row["completion"]["bookmark"]["sha256"] == bookmark["sha256"]
            and (not promoted or row["same_retained_file"] is True),
            "saved artifact",
        )
    require(
        (row["content_frames"], row["start_frame"]) == (count, start),
        "wrong source range",
    )
    capture(row, "baseline_monitor")
    confirmed(
        row, "retirement_confirmed", "final_retirement_confirmed", "owner_released"
    )
    c = row["completion"]
    if promoted or "fixture_native_mode_at_admission" in row:
        require(
            row["fixture_native_mode_at_admission"]
            == row["fixture_native_mode_after_retirement"]
            == "0o600",
            "retained file mode",
        )
    if promoted:
        fixture(
            row["fixture"],
            count,
            row["fixture_native_mode_after_retirement"],
            c["bookmark"]["sha256"],
        )
        require(
            row["fixture_native_mode_at_admission"] == "0o600", "fixture admission mode"
        )
    if case == "exact_eof":
        terminal(c, "already_at_end", "logical_eof", None, count)
        return "already_at_end_no_launch"
    quiet(row["post_reap_quiet"])
    if case == "early_zero_exit":
        terminal(c, "uncertain", "terminal_progress_unconfirmed", 1, count)
        fault = row["fault"]
        confirmed(fault, "applied", "duplicate_handle_closed")
        require(not fault.get("error"), "fault injection failed")
        require(
            fault["requested_exit_code"] == 0
            and c["evidence"]["launched_at_s"]
            <= fault["at_s"]
            <= c["evidence"]["reaped_at_s"],
            "missing fault",
        )
        prefix = match(row, "waveform_match", "waveform_diagnostics", 0)
        require(
            0 < prefix["end_source_s"] < count / RATE - TAIL
            and c["bookmark"]["frame"] < count,
            "negative control did not truncate",
        )
        return "uncertain_missing_tail"
    if case.endswith("_stop"):
        terminal(c, "stopped", "interrupted", 1, count)
        confirmed(row, "stop_acknowledged")
        e = c["evidence"]
        if case == "startup_stop":
            require(
                e["stop_at_s"] - e["launched_at_s"] <= 0.250
                and row["waveform_match"] is None
                and row["waveform_diagnostics"]["timeline_stage"] == "no_signal_packets"
                and c["bookmark"]["frame"] == start,
                "startup Stop",
            )
            return "startup_stopped"
        obs = row["stop_observation"]
        require(
            obs["generation"] == c["generation"]
            and e["launched_at_s"] <= obs["observed_at_s"] <= e["stop_at_s"]
            and e["stop_at_s"] - obs["observed_at_s"] <= 0.5,
            "stale Stop",
        )
        m = match(
            row,
            "waveform_match",
            "waveform_diagnostics",
            0,
            count / RATE if case == "padding_stop" else None,
        )
        require(m["capture_end_qpc_s"] - e["stop_at_s"] <= STOP, "late audible Stop")
        if case == "padding_stop":
            require(
                obs["frame"] == c["bookmark"]["frame"] == count
                and num(e, "last_physical_frame") > count
                and e["stop_at_s"] < e["reaped_at_s"],
                "padding phase",
            )
        else:
            require(
                0 < c["bookmark"]["frame"] < count
                and m["end_source_s"] < count / RATE - TAIL
                and abs(c["bookmark"]["frame"] / RATE - m["end_source_s"]) <= SEEK,
                "Stop bookmark alignment",
            )
        return "stopped"
    terminal(c, "natural", "normal_exit", 1, count)
    if case == "near_eof_one_frame":
        d = row["waveform_diagnostics"]
        require(
            row["waveform_match"] is None
            and d["timeline_stage"] == "contiguous"
            and d["stage"] == "invalid_capture"
            and type(row["capture_metrics"]["packet_count"]) is int
            and row["capture_metrics"]["packet_count"] > 0
            and 0 < d["content_frames"] < 0.03 * RATE
            and 0 <= num(d, "packet_clock_error_max_s") <= 0.005,
            "unexpected one-frame measurement",
        )
        return "natural_waveform_unqualified"
    match(row, "waveform_match", "waveform_diagnostics", start / RATE, count / RATE)
    return "natural_waveform_pass"


def grade_transfer(row, *, promoted=False, pilot=False):
    pair = (row["old_kind"], row["new_kind"], row["phase"])
    require(
        pair in PAIRS and not row.get("error") and not row.get("feed_unsettled"),
        "transfer failed",
    )
    capture(row, "baseline")
    confirmed(
        row,
        "old_lease_revoked",
        "old_retirement_confirmed",
        "new_retirement_confirmed",
        "owner_released",
    )
    request, grant = (
        num(row, "acquire_requested_at_s"),
        num(row, "successor_granted_at_s"),
    )
    require(0 < request <= grant and grant - request <= STOP, "transfer timing")
    lifetimes = {}
    for role, gen, count in (("old", 1, 120000), ("new", 2, 24000)):
        done = row[role + "_completion"]
        if promoted:
            fixture(
                row[role + "_fixture"],
                count,
                row[role + "_fixture_mode_after_retirement"],
                done.get("bookmark", {}).get("sha256"),
            )
        if row[role + "_kind"] == "controlled":
            old = role == "old"
            terminal(
                done,
                "stopped" if old else "natural",
                "interrupted" if old else "normal_exit",
                gen,
                count,
            )
            e = done["evidence"]
            lifetimes[role] = (
                e["launched_at_s"],
                e["retired_at_s"] if old else e["reaped_at_s"],
            )
            if old:
                require(
                    request <= e["stop_at_s"] <= e["retired_at_s"] <= grant,
                    "transfer before retirement",
                )
                if row["phase"] == "padding":
                    require(
                        done["bookmark"]["frame"] == count
                        and e["last_physical_frame"] >= count,
                        "padding trigger",
                    )
            else:
                require(grant <= e["launched_at_s"], "unauthorized successor")
        else:
            old = role == "old"
            pump = done["pump"]
            require(
                pump["outcome"] == ("stopped" if old else "drained")
                and pump["bytes_fed"] == count * 4,
                "PCM terminal",
            )
            events = row[role + "_events"]
            require(
                [name for name, _ in events]
                == ["SinkStarted", "SinkStopped" if old else "SinkDrained"]
                and events[0][1] <= events[1][1],
                "PCM evidence",
            )
            lifetimes[role] = (events[0][1], request + STOP if old else events[-1][1])
            if old:
                require(
                    done["native_retirement"] is True
                    and request <= events[-1][1] <= grant,
                    "stale PCM retirement",
                )
            else:
                require(grant <= events[0][1], "unauthorized PCM successor")
    new = match(row, "new_match", "new_diagnostics", 0, 0.5)
    if row["old_match"] is None:
        check_lifetime(new, *lifetimes["new"])
        require(
            row["phase"] == "startup"
            and row["old_diagnostics"]["timeline_stage"] == "no_signal_packets",
            "unqualified old source",
        )
        quiet(row["old_channel_monitor"])
        require(
            row["old_channel_monitor"]["packet_frames"] > 0, "unobserved old channel"
        )
        return None
    old = match(
        row,
        "old_match",
        "old_diagnostics",
        0,
        2.5 if row["phase"] == "padding" else None,
    )
    gap = new["capture_onset_qpc_s"] - old["capture_end_qpc_s"]
    require(
        abs(gap - num(row, "old_end_to_successor_onset_s")) <= 1e-9, "gap arithmetic"
    )
    require(gap >= 2 * TAIL, "overlap or insufficient edge margin")
    check_lifetime(old, *lifetimes["old"])
    check_lifetime(new, *lifetimes["new"])
    require(old["capture_end_qpc_s"] <= request + STOP, "transfer Stop observer bound")
    for field in ("left_channel_during_new_only", "right_channel_during_old_only"):
        if pilot:
            require(field not in row, "pilot evidence inventory")
            continue
        quiet(row[field])
        require(row[field]["packet_frames"] > 0, "unobserved channel isolation")
    return gap


def grade_matrix(rows, *, promoted=False):
    require(
        Counter((r["round"], r["old_kind"], r["new_kind"], r["phase"]) for r in rows)
        == Counter((n, *p) for n in (1, 2, 3) for p in PAIRS),
        "incomplete transfer matrix",
    )
    gaps = [
        gap for r in rows if (gap := grade_transfer(r, promoted=promoted)) is not None
    ]
    return {
        "transfers": len(rows),
        "matched_pairs": len(gaps),
        "quiet_old": len(rows) - len(gaps),
        "minimum_gap_s": min(gaps),
        "maximum_gap_s": max(gaps),
    }


def grade_controls(rows, *, promoted=False):
    require(
        Counter(r["case"] for r in rows)
        == Counter(k for k in CASES if k != "early_zero_exit"),
        "incomplete adapter matrix",
    )
    stopped = next(r for r in rows if r["case"] == "active_stop")
    return {r["case"]: grade_control(r, stopped, promoted=promoted) for r in rows}


def grade_negative(row, *, promoted=False):
    require(
        row["case"] == "early_zero_exit", "expected early-zero-exit negative control"
    )
    result = grade_control(row, promoted=promoted)
    require(
        result == "uncertain_missing_tail", "negative control did not detect truncation"
    )
    return result


def grade_promoted(receipt, kind):
    # This is the measurement schema of the promoted drivers, independent of the
    # current checkout's bytes. Historical pilot compatibility is archive-only.
    require(
        type(receipt["schema_version"]) is int and receipt["schema_version"] == 2,
        "expected promoted schema 2",
    )
    sources = {
        "tldw_chatbook/" + n
        for n in (
            "TTS/_controlled_wav_windows.py",
            "TTS/controlled_wav_playback.py",
            "TTS/controlled_wav_profile.py",
            "TTS/controlled_wav_progress.py",
            "TTS/audio_output_ownership.py",
            "TTS/audio_output_runs.py",
            "Audio/streaming_sink.py",
            "Utils/windows_files.py",
            "Utils/private_paths.py",
            "Utils/platform_files.py",
        )
    }
    sources.update(
        "tools/evidence/watchlist_article_audio/" + n
        for n in (
            "qualification.py",
            "wasapi_packets.py",
            "pulse_fragments.py",
            "controlled_instrument.py",
            "windows_controlled_adapter.py",
            "windows_output_transfers.py",
            "windows_early_exit.py",
        )
    )
    hashes = receipt["source_file_sha256"]
    require(
        set(hashes) == sources
        and all(
            isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
            for v in hashes.values()
        )
        and re.fullmatch("[0-9a-f]{40}", receipt["source_revision"]),
        "invalid source manifest",
    )
    require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", receipt["date"])
        and receipt["machine"]
        and receipt["python"]
        and receipt["windows_version"]
        and set(receipt["packages"])
        == {"numpy", "scipy", "cffi", "soundcard", "sounddevice"},
        "missing capture environment",
    )
    if kind == "controls":
        return grade_controls(receipt["runs"], promoted=True)
    if kind == "transfers":
        return grade_matrix(receipt["runs"], promoted=True)
    require(kind == "negative", "unknown promoted receipt kind")
    return grade_negative(receipt, promoted=True)


def grade_archive(directory):
    def read(name):
        receipt = json.loads(
            (directory / ("controlled-adapter-" + name + ".json")).read_text(
                encoding="utf-8"
            )
        )
        if name != "canonical-controls-6":
            historical = None if name in ("early-zero-exit", "quiet-preflight-1") else 1
            require(
                receipt.get("schema_version") == historical,
                "unexpected historical receipt schema",
            )
        return receipt

    for label in ("native-control-1", "native-controls-2", "native-controls-3"):
        pilot = read(label)["runs"]
        require(
            len(pilot) == (2 if label == "native-controls-3" else 1)
            and pilot[0]["case"] == "natural_500ms",
            "pilot inventory",
        )
        grade_control(pilot[0])
    controls = read("native-controls-4")["runs"]
    for r, expected, size in (
        (pilot[1], "natural_120ms", 71084),
        (controls[-1], "padding_stop", 144044),
    ):
        require(
            r["case"] == expected
            and r["prepare_refusal"] == "invalid_retained_wave"
            and r["no_launch"] is True
            and r["validation_diagnostics"]
            == [
                {
                    "exception": "PermissionError",
                    "mode": "0o644",
                    "owner_matches": True,
                    "links": 1,
                    "size": size,
                }
            ],
            "historical refusal",
        )
    for label, expected in (
        ("transfer-pilot-1", "controlled"),
        ("pcm-transfer-pilot-2", "pcm"),
    ):
        pilot = read(label)["runs"]
        require(len(pilot) == 1 and pilot[0]["old_kind"] == expected, "transfer pilot")
        grade_transfer(pilot[0], pilot=True)
    rows = [r for r in controls if not r.get("prepare_refusal")] + read(
        "padding-stop-5"
    )["runs"]
    results = grade_controls(rows)
    results["early_zero_exit"] = grade_negative(read("early-zero-exit"))
    capture(read("quiet-preflight-1"), "baseline_monitor")
    canonical = read("canonical-controls-6")
    return {
        "controls": results,
        "canonical_controls": grade_promoted(canonical, "controls"),
        **grade_matrix(read("transfers-3rounds")["runs"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--kind",
        choices=("archive", "controls", "transfers", "negative", "preflight"),
        default="archive",
    )
    args = parser.parse_args()
    if args.kind == "archive":
        result = grade_archive(args.input)
    else:
        receipt = json.loads(args.input.read_text(encoding="utf-8"))
        if args.kind == "preflight":
            capture(receipt, "baseline_monitor")
            result = "quiet"
        else:
            result = grade_promoted(receipt, args.kind)
    print(json.dumps(result, indent=2))
