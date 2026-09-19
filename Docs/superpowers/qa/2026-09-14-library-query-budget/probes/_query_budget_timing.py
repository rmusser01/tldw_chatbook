"""Disposable complete-helper timing; warm mounted production screen."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import pytest

from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)


@pytest.mark.asyncio
async def test_measure():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        focus_selector = "#library-search-input"

        def cached_rail():
            return screen._library_layout_ref("#library-rail")

        def cached_focus():
            return screen._library_layout_ref(focus_selector) in screen.focus_chain

        def chain_focus():
            return any(w.id == "library-search-input" for w in screen.focus_chain)

        measurements = {}
        for name, fn in [
            ("active_rail", screen._active_library_rail),
            ("cached_rail", cached_rail),
            ("focusable", lambda: screen._library_focusable(focus_selector)),
            ("cached_focus", cached_focus),
            ("chain_focus", chain_focus),
        ]:
            fn()
            samples = []
            for _ in range(7):
                start = perf_counter_ns()
                for _ in range(500):
                    fn()
                samples.append((perf_counter_ns() - start) / 500)
            measurements[name] = {"median_ns": median(samples), "samples_ns": samples}
        Path(
            ".superpowers/sdd/2026-09-14-library-query-budget/helper-timing-after.json"
        ).write_text(json.dumps(measurements, indent=2))
