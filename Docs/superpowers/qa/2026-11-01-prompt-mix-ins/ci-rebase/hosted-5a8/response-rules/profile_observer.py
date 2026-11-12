"""Metadata-only Python profile observer: no patched product callables."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT=Path('/private/tmp/mixins-5a8-ui26-diagnosis')
EVENTS=[]
START=time.monotonic()
NAMES={'_sync_native_console_chat_ui','_sync_console_session_draft','_resume_console_workspace_conversation','_close_console_session_tab','_send_console_message_from_visible_action_observed','_dispatch_console_command','refuse_send_from_unbound_composer','_handle_exception','_capture_console_draft_switch_snapshot','_bind_composer_to_active_session','warm','_refresh','prepare_console_presentation_sync','manage','open_manager'}


def snapshot(screen):
    if screen is None:
        return {}
    store=getattr(screen,'_console_chat_store',None)
    session=getattr(screen,'_session',screen)
    composer=getattr(screen,'_console_composer_or_none',lambda:None)()
    draft=composer.draft_text() if composer else ''
    full=getattr(screen,'_console_completed_full_source',None)
    result={name:getattr(screen,name,None) for name in ['_console_sync_in_progress','_console_sync_requested','_console_attach_sync_complete','_console_attach_reconciled','_console_attach_reconcile_running','_console_control_bar_replay_whole_sync','_closing']}
    result.update(active=getattr(store,'active_session_id',None),bound=getattr(session,'_console_visible_draft_session_id',None),sessions=list(getattr(store,'_sessions',{})),draft_kind=('/rules' if draft=='/rules' else '/omfg' if draft.startswith('/omfg') else 'text' if draft else 'empty'),draft_length=len(draft),activation=getattr(session,'_conversation_activation_depth',None),full_source_session=getattr(full.get('session'),'id',None) if isinstance(full,dict) else None)
    switch=getattr(session,'_console_draft_switch_snapshot',None)
    result['switch_owner']=switch[0] if switch else None
    try:
        button=screen.query_one('#console-send-message')
        result.update(button_object=id(button),button_disabled=button.disabled,button_active=button.has_class('-active'),button_region=str(button.region))
    except Exception:
        pass
    return result


def chains():
    out=[]
    for task in asyncio.all_tasks():
        coro=task.get_coro();frames=[];seen=set()
        while coro is not None and id(coro) not in seen:
            seen.add(id(coro));frame=getattr(coro,'cr_frame',None) or getattr(coro,'gi_frame',None)
            if frame is not None:
                frames.append(f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}')
            coro=getattr(coro,'cr_await',None) or getattr(coro,'gi_yieldfrom',None)
        out.append({'task':task.get_name(),'chain':frames})
    return out


def profile(frame,event,arg):
    if event not in {'call','return'}:
        return
    code=frame.f_code;name=code.co_name;filename=code.co_filename
    obj=frame.f_locals.get('self')
    if filename.endswith('/textual/pilot.py') and name in {'click','_post_mouse_events'}:
        app=getattr(obj,'app',None)
        screen=next((s for s in reversed(app.screen_stack) if hasattr(s,'_console_chat_store')),None) if app else None
    elif filename.endswith('/textual/widgets/_button.py') and name in {'_on_click','press','_on_mouse_down','_on_mouse_up'} and getattr(obj,'id',None)=='console-send-message':
        try:
            screen=obj.screen
        except Exception:
            screen=None
    elif name=='_handle_exception' and '/textual/app.py' in filename:
        screen=next((s for s in reversed(obj.screen_stack) if hasattr(s,'_console_chat_store')),None)
    elif name in NAMES and '/tldw_chatbook/' in filename:
        screen=frame.f_locals.get('screen') or getattr(obj,'screen',None) or getattr(obj,'_screen',None)
        if screen is None and hasattr(obj,'_console_chat_store'):
            screen=obj
    elif name=='_session_or_raise' and '/tldw_chatbook/' in filename:
        sid=frame.f_locals.get('session_id')
        if sid in getattr(obj,'_sessions',{}):
            return
        screen=None
    else:
        return
    value={'t':round(time.monotonic()-START,6),'event':event,'function':name,'file':str(Path(filename).name),'line':frame.f_lineno,'frame':id(frame),'state':snapshot(screen)}
    if name=='_handle_exception':
        value['error']=repr(frame.f_locals.get('error'))
        value['tasks']=chains()
    if name=='_session_or_raise':
        value['missing_session']=frame.f_locals.get('session_id')
        value['known_sessions']=list(getattr(obj,'_sessions',{}))
    if name in {'_on_click','press'}:
        value.update(disabled=getattr(obj,'disabled',None),active=obj.has_class('-active'),button_object=id(obj))
    if name in {'click','_post_mouse_events'}:
        value['selector']=repr(frame.f_locals.get('widget',frame.f_locals.get('selector')))
        if event=='return':
            value['return']=repr(arg)[:80]
            value['tasks']=chains()
    EVENTS.append(value)


class Observer:
    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_call(self,item):
        before=sys.getprofile()
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(before)
            (ROOT/(os.environ.get('UI26_TAG','profile')+'.events.json')).write_text(json.dumps(EVENTS,indent=2)+'\n')


if __name__=='__main__':
    raise SystemExit(pytest.main(['-q','-x','-p','no:cacheprovider','--tb=short','Tests/UI/test_response_rules_end_to_end.py::test_learn_repair_reopen_and_exclude_through_console[ancestry-size1]'],plugins=[Observer()]))
