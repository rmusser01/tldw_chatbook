"""Replay the recorded capacity model without downloading historical CI logs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main() -> None:
    """Print measured-model shard loads for this census or its original base."""
    artifact_dir = Path(__file__).resolve().parent
    repo = next(
        parent
        for parent in artifact_dir.parents
        if (parent / "scripts/ui_lane_shard_replay.py").is_file()
    )
    summary = json.loads((artifact_dir / "capacity-summary.json").read_text())
    measured = json.loads((artifact_dir / "capacity-entry-weights.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--census", type=Path, default=repo / "scripts/ui_pr_gate_census.txt"
    )
    parser.add_argument("--shards", type=int, default=summary["applied_shards"])
    parser.add_argument("--charge-multiplier", type=float, default=1.0)
    parser.add_argument("--rebalance", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "ui_replay", repo / "scripts/ui_lane_shard_replay.py"
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    entries = replay.read_census(args.census)
    text = args.census.read_text(encoding="utf-8")
    missing = [node for node in summary["new_selector_order"] if node not in entries]
    entries += missing
    text = text.rstrip("\n") + "\n" + "\n".join(missing) + "\n"
    costs = {
        node: seconds * summary["inherited_cost_factor"]
        for node, seconds in measured["entries"].items()
    }
    costs.update(
        {
            node: seconds * args.charge_multiplier
            for node, seconds in summary["new_raw_windows_seconds"].items()
        }
    )
    if len(entries) != len(set(entries)) or set(entries) != set(costs):
        raise ValueError("Use the recorded census or its base with the 27 new entries.")
    swaps = 0
    if args.rebalance:
        entries, swaps = replay.rebalanced(
            entries,
            costs,
            args.shards,
            summary["pytest_target_minutes"] * 60,
            replay.anchored(text),
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
                "meets_target": max(loads) <= summary["pytest_target_minutes"],
                "matches_recorded_order": order_hash == summary["final_order_sha256"],
                "loads_minutes": loads,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
