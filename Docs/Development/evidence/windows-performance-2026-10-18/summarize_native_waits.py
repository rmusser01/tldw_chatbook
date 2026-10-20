"""Retain every native-wait return and exact-exit prerequisite before pooling."""

from collections import Counter
import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
comparisons = {}
pooled = {"control": [], "after": []}
for tag in ("control1", "after1", "after2", "control2"):
    receipt = json.loads((OUT / f"native-waits-{tag}-exit.json").read_text())
    if receipt.get("exit_code") != 0:
        raise ValueError(f"Unsuccessful exact process exit: {tag}")
    packet = json.loads((OUT / f"plain-ready-native-waits-{tag}.json").read_text())
    if packet["blocked_network_attempts"] or packet["real_profile_refusals"]:
        raise ValueError(f"Guard events: {tag}")
    rows = packet["console_resume"]
    if len(rows) != 6 or not all(
        r["draft_retained"] and r["attach_reconciled"] for r in rows
    ):
        raise ValueError(f"Incomplete journey: {tag}")
    comparisons[tag] = {
        "returns": rows,
        "median_input_ms": median(r["resume_to_input_ms"] for r in rows),
        "median_attach_ms": median(r["resume_to_attach_ms"] for r in rows),
        "worst_input_ms": max(r["resume_to_input_ms"] for r in rows),
        "worst_attach_ms": max(r["resume_to_attach_ms"] for r in rows),
        "exit_receipt": f"native-waits-{tag}-exit.json",
        "guards": [packet["blocked_network_attempts"], packet["real_profile_refusals"]],
        "baseline_scope": packet.get("baseline_scope"),
    }
    pooled["control" if tag.startswith("control") else "after"].extend(rows)
summary = {
    "baseline_revision": "d24919ab402d503ca8210dc5ee6b0fd186adbd64",
    "baseline_scope": ["tldw_chatbook/UI/Screens/home_screen.py"],
    "runs": comparisons,
    "pooled": {
        arm: {
            "input_ms": median(r["resume_to_input_ms"] for r in rows),
            "attach_ms": median(r["resume_to_attach_ms"] for r in rows),
            "worst_input_ms": max(r["resume_to_input_ms"] for r in rows),
            "worst_attach_ms": max(r["resume_to_attach_ms"] for r in rows),
        }
        for arm, rows in pooled.items()
    },
}
(OUT / "native-waits-comparison.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary["pooled"], indent=2))

diagnostics = {}
for tag in ("discovery", "phases", "home", "verified-final", "verified-control"):
    source = OUT / f"plain-ready-native-waits-{tag}.json"
    if not source.exists():
        continue
    packet = json.loads(source.read_text())
    receipt = json.loads((OUT / f"native-waits-{tag}-exit.json").read_text())
    if receipt.get("exit_code") != 0:
        raise ValueError(f"Unsuccessful diagnostic exit: {tag}")
    rows = packet.get("native_wait_spans", [])
    edges = packet.get("native_wait_home_edges", [])
    hidden = False
    hidden_starts = 0
    for _stage, name, edge, _time in edges:
        if name.endswith("on_screen_suspend"):
            hidden = True
        if name.endswith("on_screen_resume"):
            hidden = False
        if name.endswith("_home_content_seam_call") and edge == "start" and hidden:
            hidden_starts += 1
    diagnostics[tag] = {
        "instrumented_discovery_only": True,
        "exit_code": receipt["exit_code"],
        "baseline_revision": packet.get("baseline_revision"),
        "baseline_scope": packet.get("baseline_scope"),
        "returns": packet["console_resume"],
        "guards": [packet["blocked_network_attempts"], packet["real_profile_refusals"]],
        "hidden_seam_attempts": hidden_starts,
        "content_worker_entries": sum(
            edge == "start" and name.endswith("invoke_seam_in_worker")
            for _stage, name, edge, _time in edges
        )
        if any(
            name.endswith("invoke_seam_in_worker")
            for _stage, name, edge, _time in edges
        )
        else None,
        "home_edges": edges,
        "slow_return_timeline": [
            row
            for row in packet.get("return_async_timeline", [])
            if tag == "discovery"
            and row[0] == "console-resume-2"
            and "sleep" not in row[1]
        ],
        "callback_spans": [
            r
            for r in rows
            if r[0].startswith("console-resume")
            and ("read_current" in r[1] or r[1].endswith(".<locals>.read"))
        ],
        "status_returns": {
            str(key): count
            for key, count in Counter(
                (r[6], r[7])
                for r in rows
                if r[0].startswith("console-resume")
                and r[1] == "_checked_display_status"
            ).items()
        },
        "status_spans": [
            r
            for r in rows
            if r[0].startswith("console-resume") and r[1] == "_checked_display_status"
        ],
    }
(OUT / "native-waits-diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2), encoding="utf-8"
)
