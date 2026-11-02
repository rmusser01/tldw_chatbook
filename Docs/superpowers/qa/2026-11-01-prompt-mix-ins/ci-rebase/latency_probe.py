"""Diagnostic observers only: delegate the unchanged real tour/Pilot/CSS work."""
import collections
import functools
import json
import os
from pathlib import Path
import time
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not os.environ.get('TLDW_TEST_PRIVATE_PROFILE_NODE'):
        yield
        return
    from textual.pilot import Pilot
    from textual.css.stylesheet import Stylesheet
    import tldw_chatbook
    module = item.module
    events = []
    active = [None]
    restores = []
    def patch(owner, name, replacement):
        original = getattr(owner, name)
        restores.append((owner, name, original))
        setattr(owner, name, replacement)
        return original
    press = Pilot.press
    @functools.wraps(press)
    async def measured_press(self, *keys):
        start=time.perf_counter(); active[0]=keys[0] if keys else None
        try:
            return await press(self,*keys)
        finally:
            events.append({'event':'press','key':active[0],'seconds':time.perf_counter()-start,'screen':type(self.app.screen).__name__,'css_sources':len(self.app.stylesheet.source)})
    patch(Pilot,'press',measured_press)
    pause = Pilot.pause
    @functools.wraps(pause)
    async def measured_pause(self,*args,**kwargs):
        start=time.perf_counter()
        try:return await pause(self,*args,**kwargs)
        finally:
            if active[0]:events.append({'event':'pause','key':active[0],'seconds':time.perf_counter()-start,'screen':type(self.app.screen).__name__})
    patch(Pilot,'pause',measured_pause)
    wait = module._wait_for_screen
    @functools.wraps(wait)
    async def measured_wait(pilot, expected, budget):
        start=time.perf_counter();at_entry=type(pilot.app.screen).__name__
        try:return await wait(pilot,expected,budget)
        finally:
            events.append({'event':'arrival_and_settle','key':active[0],'seconds':time.perf_counter()-start,'at_entry':at_entry,'screen':type(pilot.app.screen).__name__})
            active[0]=None
    patch(module,'_wait_for_screen',measured_wait)
    def instrument(name):
        original=getattr(Stylesheet,name)
        @functools.wraps(original)
        def observed(self,*args,**kwargs):
            start=time.perf_counter()
            try:return original(self,*args,**kwargs)
            finally:
                if active[0]:events.append({'event':'css_'+name,'key':active[0],'seconds':time.perf_counter()-start})
        patch(Stylesheet,name,observed)
    for name in ('parse','reparse'):
        instrument(name)
    try:yield
    finally:
        for owner,name,original in reversed(restores):setattr(owner,name,original)
        Path(os.environ['MIXINS_LATENCY_PROBE_OUTPUT']).write_text(json.dumps({'package':tldw_chatbook.__file__,'test':str(item.path),'css_test_cache':os.environ.get('TLDW_TEST_CSS_CACHE'),'events':events},indent=2)+'\n')
