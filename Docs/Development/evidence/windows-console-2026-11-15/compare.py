"""Validate four native packets; aggregate latencies only for complete cohorts."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--names", nargs=4, default=("paired-a2", "paired-c1", "paired-c2", "paired-a3")
)
parser.add_argument("--baseline", default="80710424ceebe3c545160fa368dfe344f9c83f78")
parser.add_argument("--candidate", default="f53f8c7bebf5fbee0148f546a1ac693054089899")
args = parser.parse_args()
names = args.names
assert all(name and all(c.isalnum() or c in "-_" for c in name) for name in names)
packets = []
commands = []
fingerprints = []
for name in names:
    stem = "plain-" + name
    report = json.loads((OUT / (stem + ".json")).read_text(encoding="utf-8"))
    receipt = json.loads((OUT / (stem + "-exit.json")).read_text(encoding="utf-8"))
    assert receipt["css_unchanged"] and report["product_sources_unchanged"], stem
    expected_head = args.baseline if name in (names[0], names[3]) else args.candidate
    assert (
        receipt["head"]
        == report["revision"]
        == report["revision_after"]
        == expected_head
    )
    command = receipt["command"].copy()
    tag_index = command.index("--tag") + 1
    assert command[tag_index] == name
    command[tag_index] = "<cohort>"
    assert all(
        flag in command
        for flag in ("--ready", "--resume-only", "--clock-safe", "--attach-poll")
    )
    assert "--return-observe" not in command and "--cpu-profile" not in command
    commands.append(command)
    assert report["diagnostic"] is False
    fingerprints.append(
        {
            key: report[key]
            for key in (
                "python",
                "platform",
                "textual",
                "diagnostic",
                "viewport",
                "attach_poll",
                "loop_clock_resolution_seconds",
                "probe_poll_seconds",
                "probe_heartbeat_seconds",
            )
        }
    )
    manifests = []
    for side in ("before", "after"):
        manifest = report[f"product_sources_{side}"]
        data = (OUT / manifest["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"], (name, side)
        manifests.append(json.loads(data))
    before, after = manifests
    assert before == after
    packets.append((name, report, receipt, before))

assert len({json.dumps(p[2]["harness_sha256"], sort_keys=True) for p in packets}) == 1
assert all(command == commands[0] for command in commands)
assert all(fingerprint == fingerprints[0] for fingerprint in fingerprints)
assert (
    len({json.dumps(p[2]["css_sha256_before"], sort_keys=True) for p in packets}) == 1
)
assert packets[0][3] == packets[3][3], "baseline source maps differ"
assert packets[1][3] == packets[2][3], "candidate source maps differ"
assert packets[0][3].keys() == packets[1][3].keys(), "product file sets differ"
changed = [p for p in packets[0][3] if packets[0][3][p] != packets[1][3][p]]
assert changed == ["tldw_chatbook/Utils/textual_css_fastpath.py".replace("/", "\\")], (
    changed
)

metrics = (
    "resume_to_input_ms",
    "resume_to_attach_ms",
    "input_ms",
    "max_loop_overshoot_ms",
)
summary = {
    "order": list(names),
    "changed_product_files": changed,
    "cohorts": {},
}
for name, report, receipt, _ in packets:
    rows = report["console_resume"]
    guards = {
        "blocked_network_attempts": report["blocked_network_attempts"],
        "real_profile_refusals": report["real_profile_refusals"],
        "observer_threads_stopped": report["observer_threads_stopped"],
        "product_sources_unchanged": report["product_sources_unchanged"],
        "css_unchanged": receipt["css_unchanged"],
    }
    qualified = (
        receipt["exit_code"] == 0
        and not receipt["timed_out"]
        and len(rows) == 6
        and not report.get("failure")
        and not report.get("pending_resume")
        and guards["observer_threads_stopped"]
        and guards["blocked_network_attempts"] == guards["real_profile_refusals"] == 0
        and all(r["draft_retained"] and r["attach_reconciled"] for r in rows)
    )
    summary["cohorts"][name] = {
        "head": receipt["head"],
        "qualified": qualified,
        "outcome": "complete" if qualified else "failed_or_incomplete",
        "exit_code": receipt["exit_code"],
        "timed_out": receipt["timed_out"],
        "completed_returns": len(rows),
        "expected_returns": 6,
        "samples": rows,
        "pending_resume": report.get("pending_resume"),
        "failure": report.get("failure"),
        "guards": guards,
    }
    if qualified:
        summary["cohorts"][name]["medians_ms"] = {
            key: statistics.median(r[key] for r in rows) for key in metrics
        }
summary["qualified"] = all(c["qualified"] for c in summary["cohorts"].values())
if summary["qualified"]:
    summary["arms"] = {}
    for arm, indexes in (("a", (0, 3)), ("b", (1, 2))):
        reports = [packets[i][1] for i in indexes]
        rows = [r for report in reports for r in report["console_resume"]]
        summary["arms"][arm] = {
            "returns": len(rows),
            "medians_ms": {
                key: statistics.median(r[key] for r in rows) for key in metrics
            },
            "worst_ms": {key: max(r[key] for r in rows) for key in metrics},
            "first_returns_ms": {
                key: [r["console_resume"][0][key] for r in reports] for key in metrics
            },
        }
summary["limits"] = [
    "Four attempted processes, six intended returns each; incomplete cohorts prevent arm aggregation.",
    "Two processes per arm are insufficient for significance or a general speed claim.",
    "Source hashing warms filesystem caches; not OS-cold or historical startup evidence.",
    "Identical stack-sampled headless journeys use a periodic traceback thread; these are not uninstrumented terminal-paint timings.",
    "The harness explicitly focuses the composer before posting its draft key; automatic arrival focus is not qualified.",
    "draft_retained means the posted key survives attachment, not that a prior draft survives navigation.",
    "The synthetic benchmark measures candidate-cache misses with warm parsed indexes and subject buckets; cold first-use construction is excluded.",
    "Instrumented discovery and earlier failed or rejected runs remain separate; this summary retains every cohort in the selected order.",
]
(OUT / "comparison.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["qualified"] else 1)
