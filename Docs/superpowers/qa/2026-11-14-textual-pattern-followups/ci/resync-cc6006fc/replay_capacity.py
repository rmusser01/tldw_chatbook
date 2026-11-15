"""Reproduce PR3569's fresh cc6006fc capacity model without GitHub log access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def main() -> None:
    """Verify the recorded union or re-run the existing swap algorithm."""
    here = Path(__file__).resolve().parent
    root = next(
        p for p in here.parents if (p / "scripts/ui_lane_shard_replay.py").is_file()
    )
    inputs = json.loads((here / "capacity-inputs.json").read_text())
    summary = json.loads((here / "capacity-summary.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--census", type=Path, default=root / "scripts/ui_pr_gate_census.txt"
    )
    parser.add_argument("--shards", type=int, default=40)
    parser.add_argument("--windows-multiplier", type=float, default=1.0)
    parser.add_argument("--rebalance", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "ui_replay", root / "scripts/ui_lane_shard_replay.py"
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    measured = inputs["measured_inherited_entry_seconds"]
    upstream = inputs["upstream111_serial_entry_seconds"]
    windows = inputs["textual27_windows_entry_seconds"]
    assert len(measured) == 2034 and len(upstream) == 111 and len(windows) == 27
    assert not set(measured) & set(upstream)
    assert not (set(measured) | set(upstream)) & set(windows)
    assert math.isclose(sum(upstream.values()), 2174.82, abs_tol=1e-8)
    assert math.isclose(sum(windows.values()), 1385.997, abs_tol=1e-8)
    costs = {
        node: seconds * inputs["upstream_cost_factor"]
        for node, seconds in (measured | upstream).items()
    } | {node: seconds * args.windows_multiplier for node, seconds in windows.items()}
    assert all(seconds > 0 for seconds in costs.values())
    entries = replay.read_census(args.census)
    missing = [
        node for node in summary["inserted_selector_order"] if node not in entries
    ]
    text = args.census.read_text(encoding="utf-8").rstrip("\n") + "\n"
    text += "\n".join(missing) + "\n"
    entries += missing
    if len(entries) != 2172 or len(set(entries)) != 2172 or set(entries) != set(costs):
        raise ValueError(
            "Use the recorded final census or the recorded upstream census."
        )
    swaps = 0
    if args.rebalance:
        entries, swaps = replay.rebalanced(
            entries, costs, args.shards, 13.0 * 60, replay.anchored(text)
        )
    loads = [
        sum(costs[node] for node in entries[index :: args.shards]) / 60
        for index in range(args.shards)
    ]
    order_hash = hashlib.sha256(("\n".join(entries) + "\n").encode()).hexdigest()
    print(
        json.dumps(
            {
                "entries": len(entries),
                "shards": args.shards,
                "swaps": swaps,
                "floor_minutes": sum(costs.values()) / 60 / args.shards,
                "maximum_minutes": max(loads),
                "minimum_minutes": min(loads),
                "meets_target": max(loads) <= 13.0,
                "matches_recorded_order": order_hash == summary["final_order_sha256"],
                "loads_minutes": loads,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
