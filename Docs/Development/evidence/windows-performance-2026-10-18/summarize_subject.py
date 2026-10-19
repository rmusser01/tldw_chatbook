"""Report every completed TASK-35613 run, with failure receipts retained."""

from collections import Counter
import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).parent
result = {}
for tag in ("control1", "after1", "after2", "control2"):
    data = json.loads((OUT / f"plain-ready-subject-{tag}.json").read_text())
    receipt = json.loads((OUT / f"subject-{tag}-exit.json").read_text())
    returns = data.get("console_resume", [])
    if len(returns) != 6 or "final_screen" not in data or receipt.get("exit_code") != 0:
        result[tag] = {"incomplete": True, "receipt": receipt}
        continue
    samples = json.loads((OUT / f"plain-ready-subject-{tag}-samples.json").read_text())
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
(OUT / "subject-comparison.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
