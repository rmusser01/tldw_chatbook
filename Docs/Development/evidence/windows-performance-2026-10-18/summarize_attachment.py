"""Summarize the complete serial attachment comparison, retaining outliers."""

import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
comparison = {"runs": {}, "pooled": {}}
pooled = {"control": [], "after": []}
for tag in ("control1", "after1", "after2", "control2"):
    report = json.loads((OUT / f"plain-ready-attachment-{tag}.json").read_text())
    receipt = json.loads((OUT / f"attachment-{tag}-exit.json").read_text())
    rows = report["console_resume"]
    assert receipt["exit_code"] == 0
    assert len(rows) == 6
    assert all(row["draft_retained"] and row["attach_reconciled"] for row in rows)
    assert report["blocked_network_attempts"] == report["real_profile_refusals"] == 0
    comparison["runs"][tag] = {
        "median_input_ms": median(row["resume_to_input_ms"] for row in rows),
        "median_attach_ms": median(row["resume_to_attach_ms"] for row in rows),
        "worst_input_ms": max(row["resume_to_input_ms"] for row in rows),
        "worst_attach_ms": max(row["resume_to_attach_ms"] for row in rows),
    }
    pooled["control" if tag.startswith("control") else "after"].extend(rows)
for arm, rows in pooled.items():
    comparison["pooled"][arm] = {
        "input_ms": median(row["resume_to_input_ms"] for row in rows),
        "attach_ms": median(row["resume_to_attach_ms"] for row in rows),
    }
comparison["improvement_percent"] = {
    name: 100
    * (1 - comparison["pooled"]["after"][name] / comparison["pooled"]["control"][name])
    for name in ("input_ms", "attach_ms")
}
text = json.dumps(comparison, indent=2)
(OUT / "attachment-comparison.json").write_text(text, encoding="utf-8")
print(text)
