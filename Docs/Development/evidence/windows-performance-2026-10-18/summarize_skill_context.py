"""Summarize retained Windows skill-context timing and main-thread samples."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

root = Path(__file__).parent
summary = {}
for variant in ("original", "fixed"):
    prefix = f"plain-ready-skills-{variant}"
    report = json.loads((root / f"{prefix}.json").read_text())
    rows = report["skill_context"]
    samples = json.loads((root / f"{prefix}-samples.json").read_text())
    main = [row for row in samples if row["stage"].startswith("skill-context-")]
    counts = Counter()
    for row in main:
        for filename, name, _line in set(map(tuple, row["stack"])):
            if "tldw_chatbook" in filename:
                counts[(Path(filename).name, name)] += row["count"]
    summary[variant] = {
        "reads": len(rows),
        "median_read_ms": median(row["elapsed_ms"] for row in rows),
        "median_max_loop_overshoot_ms": median(row["max_loop_overshoot_ms"] for row in rows),
        "max_loop_overshoot_ms": max(row["max_loop_overshoot_ms"] for row in rows),
        "main_thread_context_admission_samples": sum(
            row["count"] for row in main
            if any("skills_scope_service.py" in entry[0] and entry[1] == "get_context" for entry in row["stack"])
            and any("storage_admission.py" in entry[0] for entry in row["stack"])
        ),
        "top_overlapping_main_thread_functions": [
            [*key, value] for key, value in counts.most_common(15)
        ],
        "blocked_network_attempts": report["blocked_network_attempts"],
        "real_profile_refusals": report["real_profile_refusals"],
    }
text = json.dumps(summary, indent=2)
(root / "skill-context-comparison.json").write_text(text, encoding="utf-8")
print(text)
