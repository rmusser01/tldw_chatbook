"""Report every completed TASK-35613 run, with failure receipts retained."""

from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

OUT = Path(__file__).parent
result = {}
confirm = "--confirm" in sys.argv
matcher = "--return-matcher" in sys.argv
if matcher and confirm:
    raise ValueError("Select one comparison")
prefix = "return-matcher" if matcher else "subject"
tags = ("control1", "after1", "after2", "control2", "final", "final-control")
if matcher:
    tags = tags[:4]
if confirm:
    tags = tuple("verify-" + tag for tag in tags[:4])
for tag in tags:
    data = json.loads((OUT / f"plain-ready-{prefix}-{tag}.json").read_text())
    receipt = json.loads((OUT / f"{prefix}-{tag}-exit.json").read_text())
    returns = data.get("console_resume", [])
    if len(returns) != 6 or "final_screen" not in data or receipt.get("exit_code") != 0:
        result[tag] = {"incomplete": True, "receipt": receipt}
        continue
    samples = json.loads((OUT / f"plain-ready-{prefix}-{tag}-samples.json").read_text())
    counts = Counter()
    for row in samples:
        if row["stage"].startswith("console-resume-"):
            counts.update(
                {name: row["count"] for name in {n for _, n, _ in row["stack"]}}
            )
    result[tag] = {
        "input_ms": [row["resume_to_input_ms"] for row in returns],
        "median_input_ms": median(row["resume_to_input_ms"] for row in returns),
        "worst_input_ms": max(row["resume_to_input_ms"] for row in returns),
        "median_attach_ms": median(row["resume_to_attach_ms"] for row in returns),
        "median_max_heartbeat_ms": median(
            row["max_loop_overshoot_ms"] for row in returns
        ),
        "all_drafts_retained": all(row["draft_retained"] for row in returns),
        "all_attached": all(row["attach_reconciled"] for row in returns),
        "network_attempts": data["blocked_network_attempts"],
        "profile_refusals": data["real_profile_refusals"],
        "exit_code": receipt["exit_code"],
        "sample_counts": {
            name: counts[name]
            for name in (
                "_check_selectors",
                "_check_subject_pseudo_classes",
                "_config_file_posture",
                "_build_console_inspector_state",
                "_unread_notification_count",
                "_checked_display_status",
            )
        },
    }
if confirm or matcher:
    benchmark_tag = "verify-benchmark" if confirm else "benchmark"
    receipt = json.loads((OUT / f"{prefix}-{benchmark_tag}-exit.json").read_text())
    if receipt.get("exit_code") != 0:
        raise ValueError("Cannot summarize an incomplete or failed CSS benchmark")
    benchmark = json.loads(
        (OUT / f"plain-ready-{prefix}-{benchmark_tag}.json").read_text()
    )
    if "final_screen" not in benchmark:
        raise ValueError("CSS benchmark report was not finalized")
    value = benchmark["subject_benchmark"]
    result["benchmark"] = {
        key: value[key] for key in ("nodes", "warm_ms", "cold_ms", "style_mismatches")
    }
    result["benchmark"]["matcher"] = {
        arm: {
            "calls": row["calls"],
            "matches": row["matches"],
            "total_instrumented_ms": sum(rule["ms"] for rule in row["rules"]),
        }
        for arm, row in value["matcher"].items()
    }
    result["benchmark"]["network_attempts"] = benchmark["blocked_network_attempts"]
    result["benchmark"]["profile_refusals"] = benchmark["real_profile_refusals"]
    result["benchmark"]["exit_code"] = receipt["exit_code"]
(
    OUT / ("subject-confirmation.json" if confirm else f"{prefix}-comparison.json")
).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
