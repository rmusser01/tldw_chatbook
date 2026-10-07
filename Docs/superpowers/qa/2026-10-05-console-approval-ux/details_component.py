"""Characterize lazy first-page preparation; never a presented-frame receipt."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo))
from tldw_chatbook.UI.Console_Modules import approval_details  # noqa: E402

assert Path(approval_details.__file__).resolve().is_relative_to(repo)
arguments = (
    {"content": "x" * 1048576, "paths": [f"synthetic-{i}" for i in range(1000)]},
)
durations = []
for _ in range(41):
    started = time.perf_counter_ns()
    page = next(approval_details.iter_redacted_details(arguments))
    durations.append((time.perf_counter_ns() - started) / 1_000_000)
    assert len(page.text) <= 4096 and page.has_more
tracemalloc.start()
page = next(approval_details.iter_redacted_details(arguments))
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
warm = sorted(durations[1:])
report = {
    "component": "details_first_page_preparation",
    "boundary": "iterator_request_to_first_page_return",
    "clock": "python_perf_counter",
    "source_sha256": hashlib.sha256(
        Path(approval_details.__file__).read_bytes()
    ).hexdigest(),
    "python": platform.python_version(),
    "platform": platform.system(),
    "warm_samples": len(warm),
    "first_use_ms": durations[0],
    "median_ms": statistics.median(warm),
    "p95_ms": warm[math.ceil(0.95 * len(warm)) - 1],
    "max_ms": max(warm),
    "first_page_peak_allocation_bytes": peak,
    "native_presented_samples": 0,
    "browser_presented_samples": 0,
    "latency_targets_qualified": False,
}
print(json.dumps(report, indent=2))
