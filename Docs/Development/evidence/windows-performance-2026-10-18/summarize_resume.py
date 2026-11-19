"""Compare identical native Console resume probes; stack counts overlap."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

root = Path(__file__).parent
summary = {}
for variant in ("before", "after", "control", "verified"):
    prefix = f"plain-ready-resume-{variant}"
    report_path = root / f"{prefix}.json"
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text())
    samples = json.loads((root / f"{prefix}-samples.json").read_text())
    counts = Counter()
    native = Counter()
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
        for name in ("list_console_unseen_marks", "get_active_workspace"):
            if any(function == name for _file, function in functions):
                native[name] += row["count"]
    cycles = report["console_resume"]
    summary[variant] = {
        "cycles": len(cycles),
        "resume_to_input_median_ms": median(c["resume_to_input_ms"] for c in cycles),
        "resume_to_input_worst_ms": max(c["resume_to_input_ms"] for c in cycles),
        "worst_loop_overshoot_median_ms": median(
            c["max_loop_overshoot_ms"] for c in cycles
        ),
        "worst_loop_overshoot_ms": max(c["max_loop_overshoot_ms"] for c in cycles),
        "draft_retained_all": (
            all(c["draft_retained"] for c in cycles)
            if all("draft_retained" in c for c in cycles)
            else None
        ),
        "attach_reconciled_all": (
            all(c["attach_reconciled"] for c in cycles)
            if all("attach_reconciled" in c for c in cycles)
            else None
        ),
        "main_thread_read_samples": {
            name: native[name]
            for name in ("list_console_unseen_marks", "get_active_workspace")
        },
        "top_overlapping_resume_functions": [
            [*key, count] for key, count in counts.most_common(25)
        ],
        "blocked_network_attempts": report["blocked_network_attempts"],
        "real_profile_refusals": report["real_profile_refusals"],
    }
text = json.dumps(summary, indent=2)
(root / "resume-comparison.json").write_text(text, encoding="utf-8")
print(text)
