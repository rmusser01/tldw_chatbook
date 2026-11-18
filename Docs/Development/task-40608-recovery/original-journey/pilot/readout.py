"""Numeric G72 report; original result remains independent of optional data."""

import argparse
import hashlib
import json
from pathlib import Path


def union(intervals):
    total = 0
    end = -1
    for lo, hi in sorted(intervals):
        if hi > max(lo, end):
            total += hi - max(lo, end)
        end = max(end, hi)
    return total


def phase(core, lo, hi):
    barriers = [row for row in core["barriers"] if lo <= row["start_ns"] < hi]
    inside = [row for row in barriers if row["end_ns"] <= hi]
    overlaps = [(max(lo, row["start_ns"]), min(hi, row["end_ns"]))
                for row in core["barriers"] if row["start_ns"] < hi and row["end_ns"] > lo]
    headers = [row for row in core["headers"] if lo <= row[0] < hi]
    complete_headers = [row for row in headers if row[1] <= hi]
    groups = {}
    for row in barriers:
        if "groups" not in row:
            continue
        for name, (nodes, hidden) in row["groups"].items():
            groups.setdefault(name, []).append([nodes, hidden])
    child_counts = [row["children_at_return"] for row in barriers if "children_at_return" in row]
    return {
        "start_ns": lo, "end_ns": hi, "wall_seconds": (hi - lo) / 1e9,
        "pilot_calls_started": len(barriers), "pilot_calls_fully_inside": len(inside),
        "pilot_queued_callbacks": sum(row.get("scheduled", 0) for row in barriers),
        "pilot_calls_without_schedule_count": sum("scheduled" not in row for row in barriers),
        "pilot_child_min": min(child_counts, default=None), "pilot_child_max": max(child_counts, default=None),
        "pilot_wait_overlap_union_seconds": union(overlaps) / 1e9,
        "header_calls_started": len(headers), "header_calls_fully_inside": len(complete_headers),
        "header_complete_wall_union_seconds": union((row[0], row[1]) for row in complete_headers) / 1e9,
        "header_complete_thread_cpu_seconds": sum(row[2] for row in complete_headers) / 1e9,
        "subtree_samples": groups,
    }


def report(directory):
    metadata = json.loads((directory / "run-metadata.json").read_bytes())
    timings = json.loads((directory / "timings.json").read_bytes())
    lab = directory / "lab-observer"
    [ui_path] = list(lab.glob("ui-*.json"))
    ui = json.loads(ui_path.read_bytes())
    children = [json.loads(path.read_bytes()) for path in lab.glob("backend-*.json")]
    guards = (ui["guard_started"] and ui["guard_finished"] and not ui["blocked_destinations"]
              and ui["exit_code"] == 0 and len(children) == len(ui["backend_pids"])
              and {item["pid"] for item in children} == set(ui["backend_pids"])
              and all(item["guard_started"] and item["guard_finished"] and not item["blocked_destinations"]
                      and item["exit_code"] == 0 and item["run_id"] == ui["run_id"]
                      and item["port"] == ui["port"] and item["role"] == "backend" for item in children))
    result = {"original_exit": metadata["exitCode"], "original_status": timings["status"],
              "source_verified": timings["source_verified"], "original_guards_complete": bool(guards),
              "candidate_commit": timings["candidate_commit"], "diagnostic_complete": False,
              "independent_arm_only": True, "phases": []}
    paths = list(lab.glob("pilot-census-ui-*.json"))
    if len(paths) != 1:
        result["optional_sink_count"] = len(paths)
        return result
    raw = paths[0].read_bytes()
    result["optional_sha256"] = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    core = data.get("census")
    if core is None:
        result["targets_installed"] = False
        return result
    result["diagnostic_complete"] = (data["pid"] == ui["pid"] and not data["probe_failed"]
                                     and data["targets_installed"] and not core["failed"]
                                     and not core["truncated"] and core["unfinished"] == 0
                                     and core["header_unfinished"] == 0)
    result["callback_cpu_seconds"] = (data["bootstrap_cpu_ns"] + core["callback_cpu_ns"]) / 1e9
    result["unfinished_barriers"] = core["unfinished"]
    result["unfinished_headers"] = core["header_unfinished"]
    result["total_barriers"] = core["calls"]
    result["total_headers"] = core["header_calls"]
    for turn in ("readme", "edit"):
        rows = [row for row in ui["events"] if row.get("turn") == turn]
        starts = [row["clock_ns"] for row in rows if row["event"] == "send"]
        http = [row["clock_ns"] for row in rows if row["event"] == "http_request_received"]
        seen = [row["clock_ns"] for row in rows if row["event"] == "reply_seen"]
        persisted = [row["clock_ns"] for row in rows if row["event"] == "reply_completed"
                     and row.get("persisted") is True and (not http or row["clock_ns"] >= http[-1])]
        if len(starts) != 1 or not (persisted or seen):
            result["phases"].append({"turn": turn, "window_complete": False})
            continue
        lo, hi = starts[0], (persisted[-1] if persisted else seen[-1])
        windows = [("send_to_reply", lo, hi)]
        if http and lo <= http[0] <= http[-1] <= hi:
            windows.extend((("before_first_http", lo, http[0]),
                            ("first_to_last_http", http[0], http[-1]),
                            ("after_last_http", http[-1], hi)))
        for name, begin, end in windows:
            value = phase(core, begin, end)
            value.update(turn=turn, phase=name, window_complete=True,
                         observer_covers_window=core["origin_ns"] <= begin <= end <= core["end_ns"],
                         terminal="persisted_reply_completed" if persisted else "reply_seen")
            result["phases"].append(value)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report(args.run_directory), output, indent=2)
        output.write("\n")
