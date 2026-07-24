# test_performance_metrics_nonblocking.py
# Description: Regression coverage for task-248 (cpu_percent event-loop freeze).
"""
Task-248: passing ``interval=0.1`` to ``psutil.Process.cpu_percent()`` makes
psutil *sleep the calling thread* for 100ms to sample CPU usage over that
window, which on Textual's single-threaded event loop is a guaranteed 100ms
UI freeze for any synchronous timer callback that does it. The non-blocking
form (``interval=None``) compares against psutil's *last* call for that
process instead of blocking to sample a fresh window.

The original offender (``Widgets/performance_metrics.py``, mounted only by
the legacy Embeddings Management window) was removed with the unreachable
SearchWindow stack in task-253; this lexical guard remains to keep the same
trap out of the live call sites the task-248 audit flagged
(``Metrics/metrics_logger.py`` and ``RAG_Search/simplified/health_check.py``)
and any future ones.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook"


def test_no_blocking_cpu_percent_interval_remains_in_source_tree():
    """Lexical guard against regression: no call site anywhere under
    tldw_chatbook/ should pass a truthy numeric interval= to cpu_percent()
    (interval=None or a bare cpu_percent() call are both fine)."""
    offenders = []
    pattern = re.compile(r"cpu_percent\(\s*interval\s*=\s*0\.")
    for path in REPO_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"Blocking cpu_percent(interval=0.x) found in: {offenders}"
