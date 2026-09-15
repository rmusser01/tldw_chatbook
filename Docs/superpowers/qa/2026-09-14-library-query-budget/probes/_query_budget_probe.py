"""Disposable query attribution and resize-signature diagnosis."""

import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import pytest
from textual.dom import DOMNode

from Tests.UI import test_library_resize_focus_gates_t23025 as gates
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

OUT = Path(".superpowers/sdd/2026-09-14-library-query-budget")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "test_resize_gate_skips_library_query_work_on_non_crossing_frames",
        "test_tab_focus_path_library_query_volume_is_bounded",
    ],
)
async def test_probe(monkeypatch, name):
    calls = Counter()
    nanos = Counter()
    signatures = []
    from contextlib import contextmanager

    @contextmanager
    def instrument(fragment, counts):
        for method in ("query", "query_one"):
            real = getattr(DOMNode, method)

            def wrapper(self, *args, _real=real, **kwargs):
                frame = sys._getframe(1)
                site = ""
                for _ in range(40):
                    if frame is None:
                        break
                    if (
                        "/tldw_chatbook/" in frame.f_code.co_filename
                        and "/site-packages/" not in frame.f_code.co_filename
                    ):
                        if fragment in frame.f_code.co_filename:
                            site = f"{frame.f_code.co_name}:{frame.f_lineno}:{args[0] if args else ''}"
                        break
                    frame = frame.f_back
                enabled = counts["enabled"] and bool(site)
                start = perf_counter_ns() if enabled else 0
                try:
                    return _real(self, *args, **kwargs)
                finally:
                    if enabled:
                        counts["n"] += 1
                        calls[site] += 1
                        nanos[site] += perf_counter_ns() - start

            monkeypatch.setattr(DOMNode, method, wrapper)
        yield counts

    monkeypatch.setattr(gates, "_counting_queries", instrument)
    signature = LibraryScreen._library_resize_layout_signature

    def trace(self):
        value = signature(self)
        previous = self._library_resize_applied_signature
        if previous is not None and value is not None:
            signatures.append(
                {
                    "width": self.size.width,
                    "changes": [
                        (i, repr(a), repr(b))
                        for i, (a, b) in enumerate(zip(previous, value))
                        if a != b
                    ],
                }
            )
        return value

    monkeypatch.setattr(LibraryScreen, "_library_resize_layout_signature", trace)
    try:
        await getattr(gates, name)(gates.LibraryHarness)
    finally:
        (OUT / f"after-{name}.json").write_text(
            json.dumps(
                {"calls": calls, "query_elapsed_ns": nanos, "signatures": signatures},
                indent=2,
            )
        )
