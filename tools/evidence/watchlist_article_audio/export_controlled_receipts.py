"""Export only the historical numerical evidence schema; never copy arbitrary logs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from grade_controlled_audio import CASES

INPUTS = {
    "canonical-controls-6": "adapter-controls.json",
    "native-control-1": "adapter-controls.json",
    "native-controls-2": "adapter-controls.json",
    "native-controls-3": "adapter-controls.json",
    "native-controls-4": "adapter-controls.json",
    "padding-stop-5": "adapter-controls.json",
    "quiet-preflight-1": "quiet-preflight.json",
    "early-zero-exit": "negative-control.json",
    "transfer-pilot-1": "transfers.json",
    "pcm-transfer-pilot-2": "transfers.json",
    "transfers-3rounds": "transfers.json",
}
NESTED = set(
    """runs validation_diagnostics capture_configuration capture_metrics baseline
    baseline_monitor completion old_completion new_completion evidence bookmark pump
    stop_observation waveform_match waveform_diagnostics post_reap_quiet old_match
    new_match old_diagnostics new_diagnostics right_channel_during_old_only
    left_channel_during_new_only old_channel_monitor fault fixture""".split()
)
NUMERIC = set(
    """schema_version content_frames start_frame no_launch owner_matches links size
    sample_rate frame generation launched_at_s stop_at_s reaped_at_s retired_at_s
    last_physical_frame advanced_span_s clock_skew_s last_observed_at_s observed_at_s
    retirement_confirmed final_retirement_confirmed owner_released capture_reaped
    actual_buffer_frames requested_buffer_frames device_period_s qpc_calibration_bracket_s
    qpc_offset_s packet_count discontinuity_raw_count session_origin_discontinuity_count
    discontinuity_count timestamp_error_count poll_gap_max_s rms packet_frames
    start_source_s end_source_s capture_onset_s capture_end_s early_score tail_score
    alignment_drift_s capture_onset_qpc_s capture_end_qpc_s packet_clock_error_max_s
    active_frames capture_frames active_start_frame active_end_frame window_frames
    early_source_frame tail_source_frame early_capture_frame tail_capture_frame duration_s
    start_error_s end_error_s applied at_s requested_exit_code duplicate_handle_closed
    stop_acknowledged ordinal round acquire_requested_at_s successor_granted_at_s
    old_lease_revoked bytes_fed native_retirement new_retirement_confirmed
    old_retirement_confirmed old_end_to_successor_onset_s new_source_start_error_s
    new_source_end_error_s separation_self_check_gaps_s source_frames logical_frames physical_frames padding_frames same_retained_file""".split()
)
ENUM = {
    "case": CASES,
    "source_stop_case": {"active_stop"},
    "status": {"pass", "fail", "unqualified"},
    "evidence": {"monitored_no_packets", "monitored_packets"},
    "kind": {"already_at_end", "natural", "stopped", "uncertain"},
    "reason": {
        "",
        "logical_eof",
        "normal_exit",
        "interrupted",
        "terminal_progress_unconfirmed",
    },
    "stage": {"invalid_capture", "matched"},
    "timeline_stage": {"contiguous", "no_signal_packets"},
    "capture_error": set(),
    "prepare_refusal": {"invalid_retained_wave"},
    "mode": {"0o644"},
    "native_mode": {"0o600"},
    "fixture_native_mode_at_admission": {"0o600"},
    "fixture_native_mode_after_retirement": {"0o600"},
    "exception": {"PermissionError"},
    "old_kind": {"controlled", "pcm"},
    "new_kind": {"controlled", "pcm"},
    "phase": {"active", "startup", "padding"},
    "outcome": {"drained", "stopped"},
    "old_events": {"SinkStarted", "SinkStopped"},
    "new_events": {"SinkStarted", "SinkDrained"},
}
FIELDS = NESTED | NUMERIC | ENUM.keys() | {"date", "source_revision", "sha256"}


def project(node, key):
    if isinstance(node, dict) and key in NESTED:
        return {k: project(v, k) for k, v in node.items() if k in FIELDS}
    if isinstance(node, list):
        if key in ("runs", "validation_diagnostics"):
            if not all(isinstance(item, dict) for item in node):
                raise ValueError(f"unexpected list for {key}")
            return [project(item, key) for item in node]
        if key in ("old_events", "new_events"):
            return [
                [project(event, key), project(stamp, "at_s")] for event, stamp in node
            ]
        if key == "separation_self_check_gaps_s" and len(node) == 2:
            return [project(item, "duration_s") for item in node]
        raise ValueError(f"unexpected list for {key}")
    if node is None:
        return None
    if type(node) in (int, float, bool) and key in NUMERIC and math.isfinite(node):
        return node
    if isinstance(node, str):
        if node in ENUM.get(key, ()):
            return node
        pattern = {
            "date": r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
            "source_revision": "[0-9a-f]{10,40}",
            "sha256": "[0-9a-f]{64}",
        }.get(key)
        if pattern and re.fullmatch(pattern, node):
            return node
    raise ValueError(f"unexpected value for {key}")


def capture_manifest(data):
    from controlled_instrument import DRIVERS, RUNTIME

    files = ["tldw_chatbook/" + n for n in RUNTIME] + [
        "tools/evidence/watchlist_article_audio/" + n for n in DRIVERS
    ]
    hashes = {
        name: project(data["source_file_sha256"][name], "sha256") for name in files
    }

    def version(s):
        if not isinstance(s, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", s):
            raise ValueError("invalid version")
        return s

    if data["machine"] != "AMD64":
        raise ValueError("unexpected machine for this archive")
    return {
        "source_file_sha256": hashes,
        "machine": data["machine"],
        "windows_version": version(data["windows_version"]),
        "python": version(data["python"]),
        "packages": {
            key: version(data["packages"][key])
            for key in ("numpy", "scipy", "cffi", "soundcard", "sounddevice")
        },
    }


def export(source, output):
    for label, leaf in INPUTS.items():
        raw = (source / ("adapter-" + label) / leaf).read_bytes()
        data = json.loads(raw)
        receipt = project(data, "runs")
        if label == "canonical-controls-6":
            manifest = (source / ("adapter-" + label) / "manifest.json").read_bytes()
            original = json.loads(manifest)
            if {k: v for k, v in data.items() if k != "runs"} != original:
                raise ValueError("manifest differs from captured receipt")
            receipt.update(capture_manifest(original))
            receipt["archived_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        receipt["archived_raw_sha256"] = hashlib.sha256(raw).hexdigest()
        (output / ("controlled-adapter-" + label + ".json")).write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
