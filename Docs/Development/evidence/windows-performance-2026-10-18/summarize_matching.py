"""Summarize all matching follow-up runs without selecting favorable returns."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[3]
result = {}
names = (
    "_check_selectors",
    "_ordered_candidates",
    "_config_file_posture",
    "_build_console_inspector_state",
    "_unread_notification_count",
)
for tag in ("before", "control1", "after1", "control2", "after2", "after3", "control3"):
    path = OUT / f"plain-ready-matching-{tag}.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    returns = data.get("console_resume", [])
    if len(returns) != 6 or "final_screen" not in data:
        result[tag] = {"incomplete": True, "completed_returns": len(returns)}
        continue
    samples = json.loads(path.with_name(path.stem + "-samples.json").read_text())
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
        "sample_counts": {name: counts[name] for name in names},
    }
result["source_sha256"] = hashlib.sha256(
    (ROOT / "tldw_chatbook/Utils/textual_css_fastpath.py").read_bytes()
).hexdigest()
text = json.dumps(result, indent=2)
(OUT / "matching-comparison.json").write_text(text, encoding="utf-8")
print(text)
