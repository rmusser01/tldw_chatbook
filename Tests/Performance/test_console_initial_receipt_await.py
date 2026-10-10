"""Real-file post-repair ownership controls; original startup has its own gate."""

import pytest
from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile
_SCRIPT = r"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install();real_profile_guard.install()
from loguru import logger
logger.remove()
route,case=sys.argv[1:];assert route=='initial_receipt_await'
selector=Path(os.environ['TLDW_CONFIG_PATH']).absolute()
selector.write_text('[general]\nusers_name="receipt-leaf"\n',encoding='utf8');selector.chmod(0o600)
from tldw_chatbook import config
from tldw_chatbook.Chat import console_runtime as runtime_module
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime, _initial_receipt_preparer
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Performance._startup_receipt_await_native_gate import ReceiptAwaitNativeGate
from Tests.Performance._startup_receipt_native_gate import physical_open

async def wait(predicate):
    deadline=asyncio.get_running_loop().time()+10
    while not predicate():
        assert asyncio.get_running_loop().time()<deadline,'exact original boundary not reached'
        await asyncio.sleep(.01)

async def main():
    directory=config.get_user_data_dir()/'initial-receipt-control';directory.mkdir(mode=0o700)
    database=CharactersRAGDB(directory/'chats.sqlite',client_id='receipt-leaf')
    creator=database.get_connection();assert physical_open(creator)
    app=SimpleNamespace(chachanotes_db=database,app_config={},conversation_local_marks_service=None)
    runtime=ConsoleRuntime(app);app.console_runtime=runtime
    gate=ReceiptAwaitNativeGate(Path.cwd().absolute(),asyncio.get_running_loop(),runtime,database)
    function=ConsoleRuntime.ensure_activity_receipt_service
    original_code,original_defaults=function.__code__,function.__defaults__
    reader=function
    helper_function=ConsoleRuntime._prepare_initial_activity_receipts
    sentinel=object()
    original_reader_get=vars(function).get('__get__',sentinel)
    original_helper_get=vars(helper_function).get('__get__',sentinel)
    original_lookup=inspect.getattr_static(ConsoleRuntime,'__getattribute__')
    original_app_descriptor=inspect.getattr_static(ConsoleRuntime,'_app',sentinel)
    namespace=vars(runtime_module)
    namespace['_receipt_replacement_entries']=[]
    namespace['_receipt_lookup_entries']=[]
    code=compile('def changed(self):\n    _receipt_replacement_entries.append(True)\n    return object()\n','<declared-source-substitution>','exec')
    changed={};exec(code,namespace,changed);changed_code=changed['changed'].__code__
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='receipt-original-owner')
    asyncio.get_running_loop().set_default_executor(pool)
    blocked,started=threading.Event(),threading.Event();occupant=None;task=None;disposal=None
    try:
        gate.start()
        if case in ('custom','instance','subclass','memory','custom_db','class_wrapper','instance_wrapper','subclass_override','lookup','app_descriptor'):
            if case=='custom':ConsoleRuntime.ensure_activity_receipt_service=lambda self:object()
            elif case=='instance':runtime.ensure_activity_receipt_service=lambda:object()
            elif case=='subclass':runtime.__class__=type('CustomRuntime',(ConsoleRuntime,),{})
            elif case=='memory':app.chachanotes_db=CharactersRAGDB(':memory:',client_id='receipt-memory')
            elif case=='custom_db':app.chachanotes_db=SimpleNamespace(db_path=database.db_path)
            elif case=='class_wrapper':ConsoleRuntime.ensure_activity_receipt_service=lambda self:reader(self)
            elif case=='instance_wrapper':runtime.ensure_activity_receipt_service=lambda:reader(runtime)
            elif case=='subclass_override':runtime.__class__=type('CustomRuntime',(ConsoleRuntime,),{'ensure_activity_receipt_service':lambda self:reader(self)})
            elif case=='lookup':
                def lookup(self,name):
                    if name=='_app':namespace['_receipt_lookup_entries'].append(True)
                    return object.__getattribute__(self,name)
                ConsoleRuntime.__getattribute__=lookup
            elif case=='app_descriptor':ConsoleRuntime._app=property(lambda self:(namespace['_receipt_lookup_entries'].append(True),app)[1])
            selected=_initial_receipt_preparer(runtime)
            if case in ('memory','custom_db'):assert selected is not None and not await selected(app,require_current=lambda:None)
            else:assert selected is None
            assert not namespace['_receipt_lookup_entries']
            assert runtime._agent_runs_db is None and gate.hold is None
            if case in ('custom','instance'):assert runtime.ensure_activity_receipt_service() is not None
            elif case=='memory':assert runtime.ensure_activity_receipt_service() is None
            elif case in ('subclass','class_wrapper','instance_wrapper','subclass_override','lookup','app_descriptor'):
                gate.release.set()
                assert runtime.ensure_activity_receipt_service() is not None
        elif case=='wrong_loop':
            selected=_initial_receipt_preparer(runtime);assert selected is not None
            def foreign_loop():
                return asyncio.run(selected(app,require_current=lambda:None))
            try:await asyncio.to_thread(foreign_loop)
            except RuntimeError as error:assert error.args==('initial_receipt_owner_changed',)
            else:raise AssertionError('foreign producer loop was admitted')
            assert runtime._agent_runs_db is None and gate.hold is None
        elif case=='borrowed':
            gate.release.set()
            assert runtime.ensure_activity_receipt_service() is not None
            borrowed=runtime._agent_runs_db._held_connection();borrowed.execute('BEGIN')
            assert await _initial_receipt_preparer(runtime)(app,require_current=lambda:None)
            assert physical_open(borrowed) and borrowed.in_transaction
            assert runtime._agent_runs_db._held_connection() is borrowed
            borrowed.rollback()
        else:
            if case in ('reader_bind','helper_bind'):
                def replaced_binding(*args):
                    namespace['_receipt_replacement_entries'].append(True)
                    return lambda *args,**kwargs:object()
                if case=='reader_bind':function.__get__=replaced_binding
                else:helper_function.__get__=replaced_binding
            if case.startswith('queued_'):
                def occupy():started.set();assert blocked.wait(10)
                occupant=pool.submit(occupy);await wait(started.is_set)
            selected=_initial_receipt_preparer(runtime);assert selected is not None
            task=asyncio.create_task(selected(app,require_current=lambda:None))
            if case.startswith('queued_'):
                await wait(lambda:gate.exact_callback_queued(task))
                assert pool._work_queue.qsize() >= 1 and not blocked.is_set()
                if case=='queued_code':function.__code__=changed_code
                elif case=='queued_defaults':function.__defaults__=(object(),)
                elif case=='queued_identity':ConsoleRuntime.ensure_activity_receipt_service=changed['changed']
                blocked.set()
            else:
                await wait(gate.entered.is_set);assert gate.hold is not None and not gate.invalid
                assert physical_open(gate.held_connection)
                if case=='repeat_cancel':
                    for _ in range(3):task.cancel();await asyncio.sleep(0);assert not task.done() and not gate.reader_returned.is_set()
                elif case=='owner':app.chachanotes_db=SimpleNamespace(db_path=database.db_path)
                elif case=='marks':app.conversation_local_marks_service=object()
                elif case=='generation':runtime.generation+=1
                elif case=='body_code':function.__code__=changed_code
                elif case=='dispose':
                    disposal=asyncio.create_task(runtime.dispose())
                    await wait(lambda:runtime._disposed)
                    assert not disposal.done() and physical_open(gate.held_connection)
                gate.release.set()
            try:
                result=await task
            except asyncio.CancelledError:
                assert case=='repeat_cancel'
            except RuntimeError as error:
                assert case.startswith('queued_') or case in ('owner','marks','generation','body_code','dispose')
                assert error.args in (('initial_receipt_source_changed',),('initial_receipt_owner_changed',))
            else:assert case in ('normal','reader_bind','helper_bind') and result is True
            assert not namespace['_receipt_replacement_entries']
            if case.startswith('queued_'):assert gate.hold is None and runtime._agent_runs_db is None
            else:
                assert gate.reader_returned.is_set() and gate.held_native_retired()
                if case=='dispose':await disposal
                if case in ('owner','marks','body_code','dispose'):assert runtime._activity_receipts is None
                if case=='repeat_cancel':assert await _initial_receipt_preparer(runtime)(app,require_current=lambda:None)
    finally:
        try:
            blocked.set();gate.release.set()
            if task is not None and not task.done():await asyncio.gather(task,return_exceptions=True)
            if occupant is not None:occupant.result(timeout=10)
            ConsoleRuntime.ensure_activity_receipt_service=reader
            function.__code__=original_code;function.__defaults__=original_defaults
            if original_reader_get is sentinel:vars(function).pop('__get__',None)
            else:function.__get__=original_reader_get
            if original_helper_get is sentinel:vars(helper_function).pop('__get__',None)
            else:helper_function.__get__=original_helper_get
            ConsoleRuntime.__getattribute__=original_lookup
            if original_app_descriptor is sentinel:
                if '_app' in vars(ConsoleRuntime):delattr(ConsoleRuntime,'_app')
            else:ConsoleRuntime._app=original_app_descriptor
            runtime.__class__=ConsoleRuntime
            vars(runtime).pop('ensure_activity_receipt_service',None)
            extra=app.chachanotes_db if app.chachanotes_db is not database else None
            app.chachanotes_db=database;app.conversation_local_marks_service=None
            if disposal is not None:await asyncio.gather(disposal,return_exceptions=True)
            try:await runtime.dispose()
            finally:
                try:
                    database.close_connection()
                    if type(extra) is CharactersRAGDB:extra.close_connection()
                finally:pool.shutdown(wait=True)
        finally:
            receipt=gate.stop()
            receipt['exact_queued_callback_qualified']=gate.queued_callback_qualified
            receipt['normal_runtime_disposal']=runtime._disposed
            receipt['creator_physically_closed']=not physical_open(creator)
            (selector.parent.parent/('initial-receipt-await-'+case+'.json')).write_text(json.dumps(receipt),encoding='utf8')
            namespace.pop('_receipt_replacement_entries',None);namespace.pop('_receipt_lookup_entries',None)
        assert receipt['original_source_current'] and not receipt['invalid'] and receipt['overflow']==0,receipt
        assert receipt['global_events_zero'] and receipt['local_masks_retired'] and receipt['tool_retired'],receipt
        assert receipt['callbacks_owned'] and receipt['local_masks_owned'],receipt
        assert receipt['normal_runtime_disposal'] and receipt['creator_physically_closed'],receipt
        if case.startswith('queued_'):assert receipt['exact_queued_callback_qualified'],receipt
    assert len(network_guard.blocked_attempts())==0
    assert not real_profile_guard.take_violations()
with user_fixture_default_owner():asyncio.run(asyncio.wait_for(main(),timeout=240))
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "case",
    (
        "normal",
        "queued_code",
        "queued_defaults",
        "queued_identity",
        "body_code",
        "repeat_cancel",
        "dispose",
        "owner",
        "marks",
        "generation",
        "borrowed",
        "custom",
        "instance",
        "subclass",
        "memory",
        "custom_db",
        "wrong_loop",
        "class_wrapper",
        "instance_wrapper",
        "subclass_override",
        "lookup",
        "app_descriptor",
        "reader_bind",
        "helper_bind",
    ),
)
def test_initial_receipt_callback_ownership(tmp_path, case):
    _run(tmp_path, "initial_receipt_await", case, script=_SCRIPT, timeout=240)
