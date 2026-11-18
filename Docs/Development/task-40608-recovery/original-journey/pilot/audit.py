"""Reproduce the compact G72p record from supplied original numeric receipts."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pilot_original_readout", HERE / "readout.py")
READOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READOUT)


def digest(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def ranges(values):
    return [min(values), max(values)] if values else None


def compact_groups(rows):
    """Keep only fixed group labels and numeric ranges; groups overlap."""
    if not rows:
        return {}
    return {
        name: {
            "samples": len(rows),
            "nodes": ranges([row["groups"][name][0] for row in rows]),
            "hidden": ranges([row["groups"][name][1] for row in rows]),
        }
        for name in rows[0]["groups"]
    }


def audit(directory):
    original = READOUT.report(directory)
    metadata = json.loads((directory / "run-metadata.json").read_bytes())
    timings = json.loads((directory / "timings.json").read_bytes())
    pins = json.loads((HERE / "source-pins.json").read_bytes())
    lab = directory / "lab-observer"
    [ui_file] = list(lab.glob("ui-*.json"))
    [sink] = list(lab.glob("pilot-census-ui-*.json"))
    backend_files = sorted(lab.glob("backend-*.json"))
    ui = json.loads(ui_file.read_bytes())
    children = [json.loads(path.read_bytes()) for path in backend_files]
    data = json.loads(sink.read_bytes())
    core = data["census"]
    # No raw event list, dynamic IDs, endpoint, run token or host path is exported.
    result = {
        "schema": 1,
        "run_label": "g72p",
        "qualifying_pair": False,
        "independent_diagnostic_arm": True,
        "candidate_commit": original["candidate_commit"],
        "source_pins_match": (
            original["candidate_commit"] == pins["candidate_commit"]
            and metadata["manifestSha256"].lower() == pins["replay_manifest_sha256"]
            and metadata["originalJourneySha256"] == metadata["candidateJourneySha256"]
            == pins["sources"]["Tests/QA/test_task40608.py"]
            and digest(HERE / "readout.py")["sha256"] == pins["diagnostic_sources"]["readout.py"]
        ),
        "original_exit": original["original_exit"],
        "original_status": original["original_status"],
        "source_verified": original["source_verified"],
        "original_guards_complete": original["original_guards_complete"],
        "guard_modes_match": ui["mode"] == "guard" and all(item["mode"] == "guard" for item in children),
        "guards": {
            "ui": {key: ui[key] for key in ("guard_started", "guard_finished", "exit_code")},
            "ui_blocked_count": len(ui["blocked_destinations"]),
            "observed_backend_count": len(ui["backend_pids"]),
            "backend_receipt_count": len(children),
            "backend_blocked_count": sum(len(item["blocked_destinations"]) for item in children),
            "backends": [{key: item[key] for key in ("guard_started", "guard_finished", "exit_code")}
                         for item in children],
        },
        "diagnostic_complete": original["diagnostic_complete"],
        "targets_installed": data["targets_installed"],
        "probe_failed": data["probe_failed"],
        "census_failed": core["failed"],
        "truncated": core["truncated"],
        "unfinished_barriers": core["unfinished"],
        "unfinished_headers": core["header_unfinished"],
        "total_barriers": core["calls"],
        "total_headers": core["header_calls"],
        "all_barriers_returned_true": all(row.get("returned_true") is True for row in core["barriers"]),
        "observer_counter_cpu_seconds": original["callback_cpu_seconds"],
        "observed_thread_cpu_quantum_seconds": 0.015625,
        "receipts": {
            "run_metadata": digest(directory / "run-metadata.json"),
            "timings": digest(directory / "timings.json"),
            "ui_guard": digest(ui_file),
            "backend_guards": [digest(path) for path in backend_files],
            "census": digest(sink),
        },
        "original_numeric_readout_receipt": pins["original_numeric_readout_receipt"],
        "canonical_numeric_readout_sha256": hashlib.sha256(
            (json.dumps(original, indent=2) + "\n").encode("utf-8")
        ).hexdigest(),
        "readout_sources": {name: digest(HERE / name) for name in ("readout.py", "audit.py", "source-pins.json")},
        "turns": {},
        "phases": [],
    }
    for phase in original["phases"]:
        result["phases"].append({key: value for key, value in phase.items()
                                 if key not in ("start_ns", "end_ns", "subtree_samples")})
        if phase.get("phase") != "send_to_reply":
            continue
        turn = phase["turn"]
        rows = [row for row in core["barriers"] if phase["start_ns"] <= row["start_ns"] < phase["end_ns"]]
        chat = [row for row in rows if row.get("screen") == "ChatScreen"]
        sampled = [row for row in chat if "groups" in row]
        result["turns"][turn] = {
            "original_adjusted_seconds": timings["turns"][turn]["original_adjusted_s"],
            "original_completed_seconds": timings["turns"][turn]["completed_s"],
            "provider_request_count": timings["turns"][turn]["events"]["http_request_received"],
            "completion_endpoint": phase["terminal"],
            "chat_barriers": len(chat), "other_screen_barriers": len(rows) - len(chat),
            "chat_children": ranges([row["children"] for row in chat]),
            "chat_sample_count": len(sampled),
            "sampled_hidden_children": ranges([row["hidden"] for row in sampled]),
            "sampled_hidden_fraction": ranges([row["hidden"] / row["children"] for row in sampled]),
            "rejected_schedules": sum(row["children"] - row["scheduled"] for row in rows),
            "barriers_with_rejected_schedules": sum(row["children"] != row["scheduled"] for row in rows),
            "subtree_samples": compact_groups(sampled),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(audit(args.run_directory), stream, indent=2)
        stream.write("\n")
