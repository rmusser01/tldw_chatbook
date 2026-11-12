"""Actual threaded imports must not permanently freeze foreground cycles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Tests.UI.test_console_first_send_preimport import _run_isolated

pytestmark = pytest.mark.bootstrap_profile

_OVERLAP = r"""
import asyncio
import gc
import json
import sys
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

from tldw_chatbook.app import TldwCli
from tldw_chatbook.app_navigation import NavigationMixin
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat import console_first_send_modules as warm
from tldw_chatbook.MCP import server
from tldw_chatbook.Utils import ui_responsiveness as policy

owner, measured, original = sys.argv[1:]
entered, release, boundary = (threading.Event() for _ in range(3))
retire = threading.Event()
module = Path.home() / '_actual_warmup_overlap.py'
module.write_text('from _overlap_events import entered, release\n'
                  'entered.set()\n'
                  'assert release.wait(15), "import release timeout"\n')
sys.path.insert(0, str(module.parent))
sys.modules['_overlap_events'] = SimpleNamespace(entered=entered, release=release)
warm.FIRST_SEND_MODULES = ('_actual_warmup_overlap',)
server.prime_local_manifest_source = lambda: None
gc.unfreeze()
gc.collect(2)
policy._HEAP_REFREEZER = policy.IdleHeapRefreezer(enabled=lambda: True)
if measured == 'measured':
    policy._HEAP_REFREEZER._look()
assert (policy._HEAP_REFREEZER._baseline is not None) == (measured == 'measured')

class Host(NavigationMixin):
    _shutting_down = False
    _screen_navigation_in_progress = TldwCli._screen_navigation_in_progress
    _screen_preimport_pacing = lambda self: (0.0, 0.0)

    def __init__(self):
        self.console_runtime = ConsoleRuntime(self, canvas_enabled_reader=lambda: False)
        self.pauses = 0
        self.callbacks = []

    def _pause_between_preimports(self, gap):
        self.pauses += 1
        if self.pauses == 2:
            # This is after the real import and the original per-step freeze,
            # before the real owner park. Keep the foreground alive here.
            boundary.set()
            assert retire.wait(15), 'foreground retirement timeout'
        TldwCli._pause_between_preimports(self, gap)

    def call_from_thread(self, callback, *args):
        self.callbacks.append((callback, args))

class Cycle:
    pass

async def main():
    app = Host()
    held = asyncio.Event()
    finish = asyncio.Event()
    references = []

    async def foreground(_message=None):
        cycle = Cycle()
        cycle.self = cycle
        references.append(weakref.ref(cycle))
        gc.collect(2)  # promote while held; young collection cannot reclaim it
        held.set()
        await finish.wait()
        del cycle
        return True

    worker = threading.Thread(target=warm.preimport_first_send_modules, args=(app,), daemon=True)
    worker.start()
    try:
        assert await asyncio.to_thread(entered.wait, 15), 'actual import not entered'
        if owner == 'navigation':
            app._handle_screen_navigation_locked = foreground
            message = SimpleNamespace(target_ownership_committed=False, report_completion=lambda value: None)
            task = asyncio.create_task(app._run_admitted_screen_navigation(message))
        else:
            intent = SimpleNamespace(turn_id='early', session_id='unselected')
            app.console_runtime._register_custody(None, received_intent=intent)
            task = asyncio.create_task(foreground())
        await asyncio.wait_for(held.wait(), 15)
        if owner == 'navigation':
            assert app._screen_navigation_in_progress() and app._screen_navigation_calls
        else:
            assert app.console_runtime.has_custodied_turns()
            assert app.console_runtime._chat_controller is None
        release.set()
        assert await asyncio.to_thread(boundary.wait, 15), 'post-import boundary not reached'
        assert references[0]() is not None
        finish.set()
        await task
        if owner == 'custody':
            app.console_runtime._release_custody('early')
        assert not app._screen_navigation_in_progress()
        assert not getattr(app, '_screen_navigation_calls', {})
        assert not app.console_runtime.has_custodied_turns()
        retire.set()
        await asyncio.to_thread(worker.join, 15)
        assert not worker.is_alive(), 'warm-up worker did not retire'
        gc.collect(2)
        retained = references[0]() is not None
        gc.unfreeze()
        gc.collect(2)
        cleaned = references[0]() is None
        print('OVERLAP_JSON:' + json.dumps({'retained': retained, 'cleaned': cleaned,
                                          'callbacks': len(app.callbacks)}))
    finally:
        release.set()
        retire.set()
        finish.set()
        await asyncio.to_thread(worker.join, 15)
        gc.unfreeze()
        gc.collect(2)

if original == 'control':
    # Retain the original first-send boot policy as a causal mutant. The
    # production import remains real and held; this only reinstates its removed
    # young-collection boot freeze at the same post-import boundary.
    real_pause = Host._pause_between_preimports
    def original_pause(self, gap):
        if self.pauses == 1:
            policy.freeze_long_lived_heap('first_send_step')
        return real_pause(self, gap)
    Host._pause_between_preimports = original_pause
asyncio.run(main())
"""


@pytest.mark.parametrize("owner", ["navigation", "custody"])
@pytest.mark.parametrize("baseline", ["unmeasured", "measured"])
@pytest.mark.parametrize("arm", ["repaired", "control"])
def test_import_overlap_cycles_retire_without_boot_retention(
    tmp_path: Path, owner: str, baseline: str, arm: str
) -> None:
    """Real GC distinguishes the repair from original boot-freeze retention."""
    result = _run_isolated(tmp_path, _OVERLAP, owner, baseline, arm)
    outcome = json.loads(
        next(
            line.removeprefix("OVERLAP_JSON:")
            for line in result.stdout.splitlines()
            if line.startswith("OVERLAP_JSON:")
        )
    )
    assert outcome["cleaned"], "unfreeze/full collection did not reclaim the cycle"
    assert outcome["retained"] is (arm == "control"), (
        "first-send boot freezing retained an overlapping foreground cycle",
        outcome,
    )
