import asyncio
import functools
import json
import os
import time
from pathlib import Path

LOG = Path(os.environ['OWNERSHIP_DIAG_LOG'])
NODE = ''

def ident(obj):
    return None if obj is None else f'{type(obj).__name__}:{id(obj):x}'

def emit(event, obj=None, **data):
    task = asyncio.current_task() if _running() else None
    app = obj if hasattr(obj, '_initial_screen_pushed') else getattr(obj, 'app_instance', None) or getattr(obj, 'app', None)
    if app is not None:
        stack = getattr(app, 'screen_stack', ())
        startup = getattr(app, '_initial_screen_setup_task', None)
        data.update(stack=[ident(s) for s in stack], initial=getattr(app, '_initial_screen_pushed', None), startup=ident(startup), startup_done=startup.done() if startup else None)
    if obj is not None:
        data.update(obj=ident(obj), generation=getattr(obj, '_console_runtime_attachment_generation', None), retired=getattr(obj, '_console_view_retired', None), closing=getattr(obj, '_closing', None), reconciled=getattr(obj, '_console_attach_reconciled', None))
    with LOG.open('a') as f:
        f.write(json.dumps(dict(t=time.monotonic(), node=NODE, event=event, task=ident(task), **data), default=str)+'\n')

def _running():
    try: asyncio.get_running_loop(); return True
    except RuntimeError: return False

def wrap(cls, name):
    original = getattr(cls, name)
    if asyncio.iscoroutinefunction(original):
        @functools.wraps(original)
        async def traced(self, *args, **kwargs):
            emit(name+'.enter', self, controller=ident(getattr(self, '_controller', self)), args=[ident(x) for x in args])
            timer = None
            if name == 'start':
                def dump():
                    emit('start.pending.tasks', self, tasks=[dict(id=ident(t), done=t.done(), name=t.get_name(), frames=[f'{f.f_code.co_filename}:{f.f_lineno}:{f.f_code.co_name}' for f in t.get_stack()]) for t in asyncio.all_tasks() if any(k in str(t.get_coro()) for k in ('start','_run','prepare','submit','owned_db'))])
                timer = asyncio.get_running_loop().call_later(4.8, dump)
            try:
                result = await original(self, *args, **kwargs)
                emit(name+'.return', self, result=result if name == 'start' else ident(result))
                return result
            except BaseException as exc:
                emit(name+'.error', self, error=type(exc).__name__)
                raise
            finally:
                if timer: timer.cancel()
        setattr(cls, name, traced)
    else:
        @functools.wraps(original)
        def traced(self, *args, **kwargs):
            view = args[0] if args else None
            emit(name+'.enter', self, view=ident(view), prior=kwargs.get('prior_generation'), attached=getattr(self, '_attached_generation', None), current=ident(getattr(self, 'view', None)), view_generation=getattr(view, '_console_runtime_attachment_generation', None))
            result = original(self, *args, **kwargs)
            emit(name+'.return', self, result=result, attached=getattr(self, '_attached_generation', None), current=ident(getattr(self, 'view', None)), hook=ident(getattr(getattr(self, '_chat_controller', None), 'notify_run_outcome', None)))
            return result
        setattr(cls, name, traced)

def pytest_collection_modifyitems(session, config, items):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Chat.console_chat_start import ConsoleChatStartCoordinator
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_send_diagnostics import SendDiagnostic
    for cls, names in [(TldwCli, ('_push_initial_screen','handle_screen_navigation','push_screen')), (ConsoleRuntime, ('attach_view','detach_view','finish_view_reconciliation')), (ChatScreen, ('_reconcile_console_after_attach',)), (ConsoleChatStartCoordinator, ('start','_prepare','_run')), (ConsoleChatController, ('submit_draft','_resolve_for_send_bounded'))]:
        for name in names: wrap(cls, name)
    original = SendDiagnostic.record
    def record(self, phase, status='entered', **fields):
        emit('attempt.stage', self, attempt=self.attempt_id, phase=phase, status=status)
        return original(self, phase, status, **fields)
    SendDiagnostic.record = record

def pytest_runtest_setup(item):
    global NODE
    NODE = item.nodeid
    emit('test.setup')
