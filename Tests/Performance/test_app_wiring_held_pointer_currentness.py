"""Causal private stock captures held at their original finite native entry."""
import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

@pytest.mark.parametrize('route', ['skill', 'collections'])
def test_actual_stock_held_pointer_refuses_before_foreign_access(tmp_path, route):
    script = r'''
import asyncio, inspect, json, os, sys, threading, time
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
from pathlib import Path
from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, _ = sys.argv[1:]
root = Path(os.environ['XDG_DATA_HOME']).parent
selected = Path(os.environ['TLDW_CONFIG_PATH'])
selected.write_text('[general]\nusers_name="held-pointer-control"\n[paths]\ndata_dir="' + (root/'data').as_posix() + '"\n', encoding='utf-8')
selected.chmod(0o600)
from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
from Tests.windows_private_fixture_runner import user_fixture_default_owner
from tldw_chatbook import app as app_source, app_service_wiring as wire
from tldw_chatbook.Widgets import compact_model_bar as metadata
from tldw_chatbook.Chat import console_preparation_reads as reads_source
from tldw_chatbook.Backup_Recovery import storage_admission as storage

async def run():
    creators = OriginalPrivateAppCreators(app_source, root)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    helper = wire._skill_wiring if route == 'skill' else wire._collections_wiring
    original_pointer = helper._source_wiring
    invoke_code = next(c for c in reads_source.run_preparation_read.__code__.co_consts if isinstance(c, type(_WorkItem.run.__code__)) and c.co_name == 'invoke')
    builder = inspect.getattr_static(type(app), '_build_local_skill_trust_service') if route == 'skill' else wire._build_deferred_collections_capture_parts
    entered, release = threading.Event(), threading.Event()
    held, invalid, foreign, observations = {}, [], [], []
    task = source = request = None
    tool = None
    facts = {'route': route, 'builder_entries': 0, 'foreign_accesses': foreign, 'observations': observations}
    class Foreign:
        def __getattr__(self, name):
            foreign.append(name)
            raise AssertionError('foreign wiring read: ' + name)
    def observe(code, offset):
        frame = parent = None
        try:
            frame = sys._getframe(1)
            assert frame.f_code is code
            if code is builder.__code__:
                facts['builder_entries'] += 1
                return
            read = frame.f_locals['read']
            if read.creator is not app or held:
                return
            assert type(read) is reads_source.ConsolePreparationRead
            assert frame.f_globals is vars(reads_source)
            assert read.task is task and not read.retired.done()
            if route == 'skill':
                assert read.callback.__self__ is app and read.callback.__func__ is builder
            else:
                assert read.callback.__globals__ is vars(helper)
                assert dict(zip(read.callback.__code__.co_freevars, (cell.cell_contents for cell in read.callback.__closure__)))['request'] is request
            parent = frame.f_back
            while parent is not None and parent.f_code is not _WorkItem.run.__code__:
                parent = parent.f_back
            assert parent is not None
            item = parent.f_locals['self']
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            assert threading.current_thread() is not threading.main_thread()
            held.update(read=read, future=item.future, thread=threading.current_thread())
        except BaseException as error:
            invalid.append(type(error).__name__ + ':' + str(error))
            entered.set()
            return
        finally:
            frame = parent = None
        entered.set()
        if not release.wait(10):
            invalid.append('native_entry_release_timeout')
    def call(label, callback):
        try:
            value = callback()
        except BaseException as error:
            observations.append([label, type(error).__name__, str(error)])
        else:
            observations.append([label, value])
    async def wait(predicate):
        end = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < end, 'held_native_entry_timeout'
            await asyncio.sleep(.005)
    try:
        if route == 'skill':
            source = wire._capture_console_skill_trust_source(app, app.skills_scope_service)
            assert source is not None, 'actual_stock_source_not_captured'
            assert source.current() and source.source_current()
        for candidate in range(5, 0, -1):
            if candidate == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(candidate, 'held-wiring-pointer')
            except ValueError:
                continue
            tool = candidate
            break
        assert tool is not None and sys.monitoring.get_events(tool) == 0
        assert sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, observe) is None
        for code in (invoke_code, builder.__code__):
            assert sys.monitoring.get_local_events(tool, code) == 0
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        if route == 'skill':
            task = asyncio.create_task(app.ensure_local_skill_trust_service(_source_current=source.source_current, _read_observers=(source.runtime_reads,)))
        else:
            app._deferred_wire_collections_capture_services()
            request = app._collections_capture_setup
            task = app._collections_capture_initializer_task
            assert request is not None and request.initializer is task
        await wait(entered.is_set)
        assert not invalid and held, invalid
        assert held['future'].running() and not task.done()
        assert held['read'] in (source.runtime_reads if route == 'skill' else request.reads)
        facts['actual_stock_captured_native_held'] = True
        helper._source_wiring = Foreign()
        if route == 'skill':
            call('source_current', source.source_current)
            call('current', source.current)
        else:
            call('request_current', request.require_current)
            call('require_current', lambda: wire._require_collections_setup_current(request))
        release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        facts['task_result'] = [type(result).__name__, str(result)]
        assert held['future'].done() and held['read'].retired.done()
        assert not (source.runtime_reads if route == 'skill' else request.reads)
        facts['physical_native_retired_before_restore'] = True
        if route == 'skill':
            assert app._local_skill_trust_service is None
            assert not app._local_skill_trust_service_build_lock.locked()
        else:
            assert app.collections_capture_repository is None
            assert app._collections_capture_setup is None
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if held:
            await asyncio.gather(asyncio.wrap_future(held['future']), return_exceptions=True)
            assert held['future'].done() and held['read'].retired.done()
        helper._source_wiring = original_pointer
        if tool is not None:
            for code in (invoke_code, builder.__code__):
                sys.monitoring.set_local_events(tool, code, 0)
            sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
            sys.monitoring.free_tool_id(tool)
        await app.console_runtime.dispose()
        facts['creator_owners_retired'] = creators.retire()
        pause = storage._begin_local_pause()
        try:
            facts['global_drain'] = pause.drain(time.monotonic() + 1)
        finally:
            pause.resume()
        facts['invalid'] = invalid
        facts['network_refusals'] = len(network_guard.blocked_attempts())
        facts['profile_refusals'] = len(real_profile_guard.take_violations())
        (root/'held-pointer-result.json').write_text(json.dumps(facts, sort_keys=True), encoding='utf-8')
    assert facts['actual_stock_captured_native_held'] and facts['physical_native_retired_before_restore']
    assert facts['builder_entries'] == 0 and not foreign, facts
    assert not invalid and facts['global_drain']
    assert facts['network_refusals'] == facts['profile_refusals'] == 0
    if route == 'skill':
        assert observations == [['source_current', False], ['current', False]], facts
        assert facts['task_result'] == ['RuntimeError', 'console_skill_trust_source_changed'], facts
    else:
        assert all(row[1:] == ['_CollectionsSetupObsolete', 'collections_capture_setup_source_changed'] for row in observations), facts
        assert facts['task_result'] == ['NoneType', 'None'], facts
    print('retired and reopened; actual held stock pointer refusal passed')
with user_fixture_default_owner():
    asyncio.run(run())
'''
    _run(tmp_path, route, 'held-pointer', script=script, timeout=45)
