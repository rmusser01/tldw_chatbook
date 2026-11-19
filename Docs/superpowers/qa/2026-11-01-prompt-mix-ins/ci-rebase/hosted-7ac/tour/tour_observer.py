"""Observe the unmodified private-profile tour; no timing or readiness overrides."""
import json
import os
import time
from pathlib import Path

rows = []
def pytest_collection_modifyitems(items):
    if not os.environ.get('TLDW_TEST_PRIVATE_PROFILE_NODE'):
        return
    target = next(i.module for i in items if i.name == 'test_destination_tour_stays_under_switch_budgets')
    from textual.pilot import Pilot
    original_press = Pilot.press
    original_wait = target._wait_for_screen
    async def press(self, *keys):
        started = time.perf_counter()
        result = await original_press(self, *keys)
        rows.append({'keys': keys, 'start': started, 'press_seconds': time.perf_counter() - started})
        return result
    async def wait(pilot, expected, budget):
        started = time.perf_counter()
        result = await original_wait(pilot, expected, budget)
        row = rows[-1]
        row.update(expected=expected, arrived=result[0], wait_seconds=time.perf_counter()-started, total_seconds=time.perf_counter()-row['start'])
        return result
    Pilot.press = press
    target._wait_for_screen = wait

def pytest_sessionfinish(session, exitstatus):
    if rows:
        Path('/private/tmp/mixins-7ac-tour/measurements.json').write_text(json.dumps({'exitstatus': int(exitstatus), 'observations': rows}, indent=2)+'\n')
