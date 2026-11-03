"""Private observer only: run original exact test without altering assertions/limits."""
import asyncio
import functools
import inspect
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path('/private/tmp/mixins-5a8-ui26-diagnosis')
TAG = os.environ.get('UI26_TAG', 'current-observed')
EVENTS = []
PATCHES = []
START = time.monotonic()


def state(screen):
    if screen is None:
        return {}
    store = getattr(screen, '_console_chat_store', None)
    composer = getattr(screen, '_console_composer_or_none', lambda: None)()
    attrs = ['_console_visible_draft_session_id', '_console_sync_in_progress', '_console_sync_requested', '_console_attach_sync_complete', '_console_attach_reconciled', '_console_attach_reconcile_running', '_console_control_bar_replay_whole_sync', '_closing']
    result = {key:repr(getattr(screen,key,None)) for key in attrs}
    result.update(active=getattr(store,'active_session_id',None), sessions=list(getattr(store,'_sessions',{})), draft=composer.draft_text() if composer else None, full_source=repr(getattr(screen,'_console_completed_full_source',None)))
    session = getattr(screen,'_session',None)
    result['activation_depth'] = getattr(session,'_conversation_activation_depth',None)
    try:
        button = screen.query_one('#console-send-message')
        result.update(button_disabled=button.disabled, button_active=button.has_class('-active'), button_region=repr(button.region))
    except Exception:
        pass
    return result


def screen_for(obj):
    if hasattr(obj,'_console_chat_store'):
        return obj
    return getattr(obj,'_screen',None)


def record(kind, **values):
    EVENTS.append({'t':round(time.monotonic()-START,6),'event':kind,**values})


def task_chains():
    chains=[]
    for task in asyncio.all_tasks():
        coro=task.get_coro();frames=[];seen=set()
        while coro is not None and id(coro) not in seen:
            seen.add(id(coro));frame=getattr(coro,'cr_frame',None) or getattr(coro,'gi_frame',None)
            if frame is not None:
                frames.append(f'{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}')
            coro=getattr(coro,'cr_await',None) or getattr(coro,'gi_yieldfrom',None)
        if any('tldw_chatbook' in frame or 'pilot.py' in frame or 'screen.py' in frame for frame in frames):
            chains.append({'task':task.get_name(),'frames':frames})
    return chains


def observe(cls,name):
    original=getattr(cls,name)
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(self,*args,**kwargs):
            screen=screen_for(self)
            record(name+'.enter',state=state(screen), args=[repr(a)[:250] for a in args],kwargs=repr(kwargs)[:400])
            try:
                result=await original(self,*args,**kwargs)
                record(name+'.return',state=state(screen),result=repr(result)[:300])
                return result
            except BaseException as error:
                record(name+'.raise',state=state(screen),error=repr(error),tasks=task_chains())
                raise
    else:
        @functools.wraps(original)
        def wrapped(self,*args,**kwargs):
            screen=screen_for(self)
            record(name+'.enter',state=state(screen),args=[repr(a)[:250] for a in args],kwargs=repr(kwargs)[:400])
            try:
                result=original(self,*args,**kwargs)
                record(name+'.return',state=state(screen),result=repr(result)[:300])
                return result
            except BaseException as error:
                record(name+'.raise',state=state(screen),error=repr(error))
                raise
    PATCHES.append((cls,name,original));setattr(cls,name,wrapped)


class Observer:
    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_call(self,item):
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
        from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
        from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
        from tldw_chatbook.UI.Console_Modules.response_rules import ConsoleResponseRulesUI
        from textual.pilot import Pilot
        from textual.widgets import Button
        for cls,names in [(ChatScreen,['_sync_native_console_chat_ui','_sync_console_native_session_tabs','handle_console_send_message','_send_console_message_from_visible_action_observed','_dispatch_console_command']), (ConsoleSessionController,['_sync_console_session_draft','refuse_send_from_unbound_composer','_close_console_session_tab']), (ConsoleWorkspaceController,['_resume_console_workspace_conversation']), (ConsoleResponseRulesUI,['manage','open_manager'])]:
            for name in names:
                observe(cls,name)
        original=Pilot.click
        @functools.wraps(original)
        async def click(self,*args,**kwargs):
            app=self.app
            screen=next((s for s in reversed(app.screen_stack) if hasattr(s,'_console_chat_store')),None)
            record('pilot.click.enter',args=repr(args),state=state(screen),tasks=task_chains())
            result=await original(self,*args,**kwargs)
            record('pilot.click.return',args=repr(args),result=result,state=state(screen),tasks=task_chains())
            return result
        PATCHES.append((Pilot,'click',original));Pilot.click=click
        original_press=Button.press
        @functools.wraps(original_press)
        def press(self,*args,**kwargs):
            if self.id=='console-send-message':
                record('button.press',disabled=self.disabled,active=self.has_class('-active'),screen=repr(self.screen))
            return original_press(self,*args,**kwargs)
        PATCHES.append((Button,'press',original_press));Button.press=press
        try:
            yield
        finally:
            for cls,name,original in reversed(PATCHES):
                setattr(cls,name,original)
            (ROOT/(TAG+'.events.json')).write_text(json.dumps(EVENTS,indent=2)+'\n')


if __name__=='__main__':
    sys.exit(pytest.main(['-q','-x','-p','no:cacheprovider','--tb=short','Tests/UI/test_response_rules_end_to_end.py::test_learn_repair_reopen_and_exclude_through_console[ancestry-size1]'],plugins=[Observer()]))
