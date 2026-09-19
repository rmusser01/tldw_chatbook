"""Observe one actual Windows initial screen barrier under a finite ceiling."""

import sys
from contextlib import asynccontextmanager


@asynccontextmanager
async def initial_screen_observation(app):
    if sys.platform != "win32":
        yield
        return

    from textual.pilot import Pilot

    original = Pilot._wait_for_screen
    consumed = False

    async def initial(pilot, *args, **kwargs):
        nonlocal consumed
        if pilot.app is not app or consumed or args or "timeout" in kwargs:
            return await original(pilot, *args, **kwargs)
        consumed = True
        try:
            return await original(pilot, timeout=135, **kwargs)
        finally:
            Pilot._wait_for_screen = original

    Pilot._wait_for_screen = initial
    try:
        yield
    finally:
        Pilot._wait_for_screen = original
