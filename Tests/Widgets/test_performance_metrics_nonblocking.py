# test_performance_metrics_nonblocking.py
# Description: RED-first regression coverage for task-248 (cpu_percent event-loop freeze).
"""
Task-248: ``PerformanceMetricsWidget._update_metrics`` called
``psutil.Process.cpu_percent(interval=0.1)`` inside a synchronous
``set_interval(2.0)`` callback -- ``interval=0.1`` makes psutil *sleep the
calling thread* for 100ms to sample CPU usage over that window, which on
Textual's single-threaded event loop is a guaranteed 100ms UI freeze every
2 seconds while Embeddings Management is open. The non-blocking form
(``interval=None``) compares against psutil's *last* call for that process
instead of blocking to sample a fresh window -- the tradeoff is that the
very first call returns 0.0/meaningless (no prior sample to diff against),
which is fine here since the widget polls every 2s indefinitely.

Also covers the two "latent copies" the audit flagged (no UI callers today,
but the same trap): ``Metrics/metrics_logger.py`` and
``RAG_Search/simplified/health_check.py``.
"""

import pathlib
import re
from unittest.mock import MagicMock

from tldw_chatbook.Widgets.performance_metrics import PerformanceMetricsWidget

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook"


def _spy_process():
    spy = MagicMock()
    spy.cpu_percent.return_value = 12.3
    spy.memory_info.return_value = MagicMock(rss=1024 * 1024 * 50)
    spy.memory_percent.return_value = 5.0
    return spy


def test_update_metrics_calls_cpu_percent_non_blocking():
    """The widget's 2s timer callback must never pass a blocking interval."""
    widget = PerformanceMetricsWidget()
    spy = _spy_process()
    widget.process = spy

    # _update_metrics() also drives _update_display()/_check_alerts(), which
    # query_one() into a DOM this bare (unmounted) widget doesn't have; that
    # failure is caught internally and logged (see the method's own
    # try/except), so it doesn't affect the cpu_percent() call under test.
    widget._update_metrics()

    spy.cpu_percent.assert_called_once_with(interval=None)


def test_update_metrics_still_updates_cpu_usage_reactive():
    """Non-blocking form must not silently stop metrics from updating."""
    widget = PerformanceMetricsWidget()
    spy = _spy_process()
    widget.process = spy

    widget._update_metrics()

    assert widget.cpu_usage == 12.3
    assert len(widget.history) == 1


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
