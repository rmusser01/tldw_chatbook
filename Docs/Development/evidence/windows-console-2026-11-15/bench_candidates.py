"""Isolate candidate-cache misses; no app, I/O, or terminal-latency claim."""

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from textual.css.stylesheet import Stylesheet  # noqa: E402
from tldw_chatbook.Utils import textual_css_fastpath as fast  # noqa: E402

tag = sys.argv[1]
assert tag in ("baseline", "candidate")
sheet = Stylesheet()
sheet.add_source(".a, .b, .c { color: red; }\n" * 500)
node = SimpleNamespace(_selector_names={".a", ".b", ".c"}, ancestors=())
expected = [id(rule) for rule in sheet.rules]
assert [id(rule) for rule in fast._ordered_candidates(sheet, node)] == expected
cache = getattr(sheet, fast._CACHE_ATTR)[3]
rounds = []
for _ in range(9):
    started = time.perf_counter_ns()
    for _ in range(100):
        cache.clear()  # Measure only misses; retain parsed rules/requirements.
        assert [id(rule) for rule in fast._ordered_candidates(sheet, node)] == expected
    rounds.append((time.perf_counter_ns() - started) / 100 / 1_000_000)
source = ROOT / "tldw_chatbook/Utils/textual_css_fastpath.py"
report = {
    "head": subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "rules": 500,
    "indexes_per_rule": 3,
    "misses_per_round": 100,
    "round_mean_ms": rounds,
    "median_round_mean_ms": statistics.median(rounds),
    "scope": "Synthetic cold candidate union and existing filter/sort; not application latency.",
}
Path(__file__).with_name(f"candidate-bench-{tag}.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report))
