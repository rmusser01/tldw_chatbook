"""Summarize native FULL/config resume runs and overlapping main-thread samples."""

from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

root = Path(__file__).parent
workload = sys.argv[1] if len(sys.argv) > 1 else "full"
summary = {}
for variant in ("before", "after", "control", "control2", "verified", "final"):
    prefix = f"plain-ready-{workload}-{variant}"
    path = root / f"{prefix}.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    cycles = report.get("console_resume", [])
    if len(cycles) != 6 or "input_timeout" in report:
        summary[variant] = {
            "completed_cycles": len(cycles),
            "excluded": "incomplete run; see native log and diagnostic report",
            "blocked_network_attempts": report.get("blocked_network_attempts"),
            "real_profile_refusals": report.get("real_profile_refusals"),
        }
        continue
    samples = json.loads((root / f"{prefix}-samples.json").read_text())
    counts = Counter()
    native = 0
    for row in samples:
        if not row["stage"].startswith("console-resume-"):
            continue
        functions = {
            (Path(filename).name, name)
            for filename, name, _line in row["stack"]
            if "tldw_chatbook" in filename
        }
        for key in functions:
            counts[key] += row["count"]
        if ("config_sync.py", "run_console_config_sync") in functions and (
            "config_participants.py",
            "operation",
        ) in functions:
            native += row["count"]
    summary[variant] = {
        "cycles": len(cycles),
        "resume_to_input_median_ms": median(c["resume_to_input_ms"] for c in cycles),
        "resume_to_input_worst_ms": max(c["resume_to_input_ms"] for c in cycles),
        "resume_to_attach_median_ms": (
            median(c["resume_to_attach_ms"] for c in cycles)
            if all("resume_to_attach_ms" in c for c in cycles)
            else None
        ),
        "resume_to_attach_worst_ms": (
            max(c["resume_to_attach_ms"] for c in cycles)
            if all("resume_to_attach_ms" in c for c in cycles)
            else None
        ),
        "worst_loop_overshoot_median_ms": median(
            c["max_loop_overshoot_ms"] for c in cycles
        ),
        "worst_loop_overshoot_ms": max(c["max_loop_overshoot_ms"] for c in cycles),
        "draft_retained_all": all(c["draft_retained"] for c in cycles),
        "attach_reconciled_all": all(c["attach_reconciled"] for c in cycles),
        "native_presentation_config_samples": native,
        "top_overlapping_resume_functions": [
            [*key, count] for key, count in counts.most_common(20)
        ],
        "blocked_network_attempts": report["blocked_network_attempts"],
        "real_profile_refusals": report["real_profile_refusals"],
    }
text = json.dumps(summary, indent=2)
(root / f"{workload}-comparison.json").write_text(text, encoding="utf-8")
print(text)
