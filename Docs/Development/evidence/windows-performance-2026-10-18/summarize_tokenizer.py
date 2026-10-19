"""Summarize native Windows Console profiles; samples are overlapping stacks."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

root = Path(__file__).parent
summary = {}
for variant in ("before", "after", "post-pr"):
    prefix = f"plain-ready-tokenizer-{variant}"
    if not (root / f"{prefix}.json").exists():
        continue
    report = json.loads((root / f"{prefix}.json").read_text())
    samples = json.loads((root / f"{prefix}-samples.json").read_text())
    counts = Counter()
    for row in samples:
        if row["stage"] not in {"idle", "typing", "return-console"}:
            continue
        for filename, name, _line in set(map(tuple, row["stack"])):
            if "tldw_chatbook" in filename:
                counts[(Path(filename).name, name)] += row["count"]
    summary[variant] = {
        "typing_median_ms": median(report["typing"]),
        "typing_worst_ms": max(report["typing"]),
        "idle_worst_loop_overshoot_ms": max(report["idle"]["loop_overshoot_ms"]),
        "idle_cpu_seconds": report["idle"]["cpu_seconds"],
        "idle_elapsed_seconds": report["idle"]["elapsed_seconds"],
        "main_thread_tokenizer_loading_samples_all_stages": sum(
            row["count"] for row in samples
            if any(
                Path(filename).name == "custom_tokenizers.py" and name in {"get_tokenizer_manager", "__init__", "load_tokenizer"}
                or Path(filename).name == "token_counter.py" and name == "get_tiktoken_encoding"
                for filename, name, _line in row["stack"]
            )
        ),
        "top_overlapping_console_functions": [[*key, count] for key, count in counts.most_common(25)],
        "blocked_network_attempts": report["blocked_network_attempts"],
        "real_profile_refusals": report["real_profile_refusals"],
        "tokenizer_display": report.get("tokenizer_display"),
    }
text = json.dumps(summary, indent=2)
(root / "tokenizer-comparison.json").write_text(text, encoding="utf-8")
print(text)
