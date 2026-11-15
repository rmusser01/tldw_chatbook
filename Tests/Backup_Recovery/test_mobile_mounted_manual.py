"""SOURCE-ONLY mounted manual Console qualification; controller review before run.

The real-home parent reaps native fixture seeding, launches a fresh public
producer, then a separate actual TldwCli/run_test child. No child is launched merely by importing this module.
This is scripted provider composition, not real provider or hardware evidence.
"""

import hashlib
import json
import os
import pwd
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from Tests.Backup_Recovery.test_mobile_late_budget_recovery import _COMMON

_PROFILE_APPEND = (
    "\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
    '[tldw_api]\nbase_url=""\napi_url=""\nurl=""\n'
    "[model_catalog]\nauto_refresh_enabled=false\n"
    "[console]\nagent_runtime=true\n"
    '[chat_defaults]\nprovider="llama_cpp"\nmodel="local-model"\n'
    '[api_settings.llama_cpp]\napi_url="http://127.0.0.1:9099"\nmodel="local-model"\n'
)


def _closed_seed_file(path):
    """Fingerprint one closed, private fixture leaf without opening a DB owner."""
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
    assert info.st_nlink == 1 and not info.st_mode & 0o077
    facts = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode,
        "size": info.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    after = path.lstat()
    assert (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    return facts


def _closed_seed_files(fixture):
    source = fixture / "profile/data/shared/mobile"
    return {
        "selector": _closed_seed_file(fixture / "profile/config.toml"),
        "source": {
            path.name: _closed_seed_file(path)
            for path in sorted(source.iterdir())
            if path.is_file()
        },
        "targets": {
            str(fixture / "profile/new/nested" / name): _closed_seed_file(
                fixture / "profile/new/nested" / name
            )
            for name in ("chat.db", "workspaces.db", "agent_runs.db")
        },
    }


def _read_seed_record(fixture, expected_digest=None):
    """Reject missing/changed seed evidence before the producer can overwrite config."""
    path = fixture / "native-seed.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None:
        assert digest == expected_digest, "seed_record_changed"
    record = json.loads(raw)
    assert record["complete"] is True and record["fixture"] == str(fixture)
    assert record["seed_calls"] == [True, False]
    assert set(record["seeded"]) == {"legacy", "canonical"}
    assert all(isinstance(value, dict) and value for value in record["seeded"].values())
    assert not (fixture / "control").exists()
    assert record["files"] == _closed_seed_files(fixture), "seed_files_changed"
    assert (fixture / "profile/config.toml").read_bytes() == bytes.fromhex(
        record["selected_config"]
    )
    assert record["seeded"]["legacy"]["fingerprints"] == {
        name: facts["sha256"] for name, facts in record["files"]["source"].items()
    }
    retired = json.loads(path.with_name("native-seed-exit.json").read_text())
    assert retired["entry_home"] == retired["exit_home"]
    assert not retired["profile_guard"] and not retired["network_guard"]
    assert retired["target_files"] == record["files"]["targets"]
    return record, digest


_SEED = r"""
import pytest
from Tests.Backup_Recovery import mobile_consolidation_fixture
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
from Tests.Backup_Recovery.test_mobile_mounted_manual import _PROFILE_APPEND, _closed_seed_files

from tldw_chatbook.Chat.console_semantic_revision import install_semantic_coordination

install_semantic_coordination()
assert not SELECTOR.exists() and not SOURCE.exists() and not CONTROL.exists()
seeded = {}
seed_calls = []
class SeedComplete(Exception):
    pass

def seed(paths, *, legacy, **options):
    assert legacy is (not seed_calls) and seed_calls in ([], [True])
    assert paths.paths == (tuple(SOURCE / name for name in ('chat.db', 'workspaces.db', 'runs.db')) if legacy else TARGETS)
    assert options == ({'artifact_root': None} if legacy else {})
    if legacy:
        # Match the original fixture config, before any binding or source review.
        with SELECTOR.open('a') as stream:
            stream.write(_PROFILE_APPEND)
    result = _small_native_seed(paths, legacy=legacy, source_case='consumers_privacy', **options)
    seeded['legacy' if legacy else 'canonical'] = result
    seed_calls.append(legacy)
    if not legacy:
        # Both native constructors returned and closed their real DB owners.
        # Stop before prepared() creates control/binds/reviews this profile.
        report_path.write_text(json.dumps({'complete': True, 'fixture': str(FIXTURE),
            'seed_calls': seed_calls, 'seeded': seeded, 'files': _closed_seed_files(FIXTURE),
            'selected_config': SELECTOR.read_bytes().hex()}, indent=2))
        raise SeedComplete
    return result
patcher=pytest.MonkeyPatch()
patcher.setattr(mobile_consolidation_fixture,'_seed',seed)
mask=os.umask(0o077)
try:
    voice_parent=FIXTURE/'.config/tldw_cli'
    voice_parent.mkdir(mode=0o700,parents=True)
    assert voice_parent.stat().st_mode & 0o777 == 0o700
    assert all(not path.exists() for path in VOICE_PATHS.values())
    try:
        with _capture_fixture(FIXTURE,patcher):
            raise AssertionError('seed unexpectedly reached prepared binding')
    except SeedComplete:
        assert seed_calls == [True, False] and not CONTROL.exists()
finally:
    os.umask(mask)
"""

_PRODUCE = r"""
import pytest
from Tests.Backup_Recovery import mobile_consolidation_fixture
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_mounted_manual import _PROFILE_APPEND, _closed_seed_files, _read_seed_record
# Validate BEFORE prepared() rewrites its base config, not after repairing it.
seed_record, seed_digest = _read_seed_record(FIXTURE, sys.argv[3])
seeded = seed_record['seeded']
seed_calls = []
def use_existing_seed(paths, *, legacy, **options):
    assert legacy is (not seed_calls) and seed_calls in ([], [True])
    assert paths.paths == (tuple(SOURCE / name for name in ('chat.db', 'workspaces.db', 'runs.db')) if legacy else TARGETS)
    assert options == ({'artifact_root': None} if legacy else {})
    if legacy:
        # The original base write plus the same append must restore exact bytes.
        with SELECTOR.open('a') as stream:
            stream.write(_PROFILE_APPEND)
    assert SELECTOR.read_bytes() == bytes.fromhex(seed_record['selected_config'])
    assert _closed_seed_files(FIXTURE) == seed_record['files']
    seed_calls.append(legacy)
    return seeded['legacy' if legacy else 'canonical']
patcher=pytest.MonkeyPatch()
patcher.setattr(mobile_consolidation_fixture,'_seed',use_existing_seed)
mask=os.umask(0o077)
try:
    voice_parent=FIXTURE/'.config/tldw_cli'
    assert voice_parent.is_dir() and not voice_parent.is_symlink()
    assert voice_parent.stat().st_mode & 0o777 == 0o700
    assert all(not path.exists() for path in VOICE_PATHS.values())
    with _capture_fixture(FIXTURE,patcher,reuse_seeded_profile=True) as values:
        _,journal,actual_selector,actual_stores,_,_,_=values
finally:
    os.umask(mask)
assert seed_calls == [True, False]
assert actual_selector==SELECTOR and actual_stores==stores and journal.root.parent==CONTROL
credentials._credential_store=lambda:fixture_credentials
assert_fixture_locators()
initial_binding=binding_facts()
assert initial_binding['pending']==[]
before=fingerprints(source)
target_before={str(path):{'device':path.stat().st_dev,'inode':path.stat().st_ino,
               'mode':path.stat().st_mode,'size':path.stat().st_size,
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in stores.paths}
_TraceReadBudget.__init__=issued
service=RecoveryService(CONTROL)
report={'scope':'fresh public producer before separate cold mounted App', 'entry_home':ENTRY_HOME,
        'guard_roots':GUARD_ROOTS,'seeded':seeded,'initial_binding':initial_binding,'target_before':target_before}
try:
    preview=service.preview_mobile_consolidation(selector,selector)
    assert preview.has_data and not preview.issues
    operation=service.start_mobile_consolidation(preview,destination_confirmed=True,
                                              rollback_password=b'test-private-mobile-rollback')
    state=service.wait(operation)
    report.update(operation=operation,source_fingerprint=preview.source_fingerprint,
                  state={k:state[k] for k in ('state','phase','issues')},
                  errors=errors,pairs_after_wait=pairs())
    assert state['state']=='succeeded',report
    assert service.pending_operations()==()
finally:
    service.close()
    report.update(source_before=before,source_after=fingerprints(source),
                  pairs_after_close=pairs(),profile_guard=take_violations(),
                  network_guard=blocked_attempts(),exit_home={k:os.environ.get(k) for k in ENTRY_HOME})
    report_path.write_text(json.dumps(report,indent=2))
assert report['source_before']==report['source_after']
assert report['exit_home']==ENTRY_HOME and not report['profile_guard'] and not report['network_guard']
assert len(report['pairs_after_close'])==1 and report['pairs_after_close'][0]['coupled']
assert require_mobile_cutover(source,stores) is None
report['claim']=claim_facts(operation)
report['final_binding']=binding_facts()
report['profile_guard']=take_violations()
report['network_guard']=blocked_attempts()
report_path.write_text(json.dumps(report,indent=2))
assert not report['profile_guard'] and not report['network_guard']
"""

_APP = r"""
import asyncio
previous=json.loads((FIXTURE/'native-producer.json').read_text())
assert previous['state']['state']=='succeeded'
assert_fixture_locators()
assert binding_facts()['pending']==[]
assert claim_facts(previous['operation'])==previous['claim']
assert require_mobile_cutover(source,stores) is None
before=fingerprints(source)
# Existing offline device-unavailable seams: no real output or microphone.
# No optional service, native owner, App lifecycle or runtime is replaced.
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
from tldw_chatbook.Audio import streaming_sink
streaming_sink._import_sounddevice=lambda:None
# Explicit proposed default-install tokenizer fallback, matching the existing
# UI fixture. Controller must inspect this scope before any App execution.
from tldw_chatbook.Utils import token_counter
assert not token_counter._ESTIMATE_CACHE
# Native no-custom-tokenizer tier, as in Tests/Chat/test_token_counter.py.
token_counter.CUSTOM_TOKENIZERS_AVAILABLE=False
token_counter.TIKTOKEN_AVAILABLE=False
token_counter.get_tiktoken_encoding=lambda model:None
from Tests.Backup_Recovery.test_mobile_owning_transactions import _ordinary_imported_owner
from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
from tldw_chatbook.DB.automatic_work import AutomaticWorkRefused
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

operation=previous['operation']
with _ordinary_imported_owner(stores,'db.agent_runs') as runs:
    with runs.transaction() as connection:
        _,children=_read_owner_receipt(connection,operation,previous['source_fingerprint'],
                                      'runs',_TraceReadBudget())
    mapped={(row['data']['domain'],row['data']['original']):row['data']['mapped']
            for row in children if row['kind']=='identity'
            and type(row['data']['original']) in (str,int)}
    original=previous['seeded']['legacy']
    ids={name:mapped[domain,original[name]] for name,domain in
         (('goal','goal'),('run','run'),('definition','agent_definition'))}
    goal=runs.goal_runs.get(ids['goal'])
    ids['chain']=goal.chain_id
    assert goal.request.objective=='Repair fixture'
    try:runs.goal_runs.set_provisioning(goal,status='ready')
    except AutomaticWorkRefused as error:assert str(error)=='import_review_required'
    else:raise AssertionError('imported Goal obtained automatic authority')

def historical(db):
    # Fixed identity-scoped reads through the actual ordinary native connection.
    # New manual rows and runtime lease tokens may legitimately differ.
    with db.transaction() as connection:
        result={
          'goal':[tuple(r) for r in connection.execute('SELECT * FROM goal_runs WHERE id=?',(ids['goal'],))],
          'chain':[tuple(r) for r in connection.execute('SELECT * FROM automatic_work_chains WHERE id=?',(ids['chain'],))],
          'reservations':[tuple(r) for r in connection.execute('SELECT * FROM automatic_work_reservations WHERE chain_id=? ORDER BY id',(ids['chain'],))],
          'holds':[tuple(r) for r in connection.execute('SELECT * FROM mobile_import_authority_holds WHERE operation_id=? ORDER BY authority_kind,authority_id',(operation,))],
          'run':[tuple(r) for r in connection.execute('SELECT * FROM agent_runs WHERE id=?',(ids['run'],))],
          'definition':[tuple(r) for r in connection.execute('SELECT * FROM agent_definitions WHERE id=?',(ids['definition'],))],
        }
        uncertain=connection.execute('SELECT state,actual_amount FROM automatic_work_reservations WHERE chain_id=?',(ids['chain'],)).fetchall()
        assert [tuple(r) for r in uncertain]==[('uncertain',None)]
        assert result['holds'] and result['goal'] and result['chain']
        return result
with _ordinary_imported_owner(stores,'db.agent_runs') as runs:
    original_rows=historical(runs)

from Tests.UI.test_console_native_chat_flow import _ReadyResolutionGateway
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
native_tool_dispatch=ToolCatalogRegistry.invoke_by_name
tool_dispatches=[]
def observe_tool_dispatch(registry,name,args,**options):
    # Observe the real native dispatch boundary and forward it unchanged.
    # No result, denial, permission, tool catalogue, or argument is substituted.
    tool_dispatches.append(name)
    return native_tool_dispatch(registry,name,args,**options)
ToolCatalogRegistry.invoke_by_name=observe_tool_dispatch
# Observe only the genuine Agents guard's first error construction. No source
# resolution, admission call, activation decision, or exception is substituted.
from tldw_chatbook.Agents import activation as agent_activation
import traceback
agent_admission_failures=[]
native_agent_error=agent_activation._guard.error
def observe_agent_error():
    error=native_agent_error()
    if not agent_admission_failures:
        context=sys.exception()
        frame=sys._getframe(1)
        owner=frame.f_locals.get('owner')
        path=frame.f_locals.get('path')
        active=frame.f_locals.get('active')
        agent_admission_failures.append({
            'native_error_type':type(error).__name__,
            'caller':{'file':frame.f_code.co_filename,'line':frame.f_lineno,'name':frame.f_code.co_name},
            'owner':owner if isinstance(owner,str) else None,
            'path':str(path) if isinstance(path,Path) else None,
            'context_type':type(context).__name__ if context is not None else None,
            'context_frames':[{'file':row.filename,'line':row.lineno,'name':row.name}
                              for row in traceback.extract_tb(context.__traceback__)] if context is not None else [],
            'active_present':active is not None,
        })
    return error
agent_activation._guard.error=observe_agent_error
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from textual.widgets import Button
class Gateway(_ReadyResolutionGateway):
    def __init__(self):self.resolutions=[];self.generations=[];self.closed=False
    async def resolve_for_send(self,selection):
        self.resolutions.append((selection.provider,selection.explicit_model))
        return await super().resolve_for_send(selection)
    async def stream_chat(self,resolution,messages,**kwargs):
        self.generations.append((resolution.provider,resolution.model))
        yield 'Exact fresh manual response Ω'
    async def aclose(self):self.closed=True

gateway=Gateway()
report={'scope':'actual mounted App/new manual Console with scripted gateway; not network/hardware',
        'entry_home':ENTRY_HOME,'guard_roots':GUARD_ROOTS,'ids':ids,'operation':operation}
async def until(predicate,pilot,seconds=60):
    deadline=asyncio.get_running_loop().time()+seconds
    while not predicate():
        assert asyncio.get_running_loop().time()<deadline,'bounded mounted transition timed out'
        await pilot.pause()
        await asyncio.sleep(0.02)

async def main():
    app=TldwCli()
    assert app.chachanotes_db.db_path==stores.chat_path
    assert app.app_config.get('console',{}).get('agent_runtime',True) is True
    assert app.app_config['chat_defaults']['provider']=='llama_cpp'
    app.console_provider_gateway_factory=lambda:gateway
    runtime=None
    try:
        async with app.run_test(size=(170,48)) as pilot:
            await until(lambda:getattr(app,'_ui_ready',False),pilot)
            await pilot.press('ctrl+2')
            await until(lambda:type(app.screen).__name__=='ChatScreen',pilot,30)
            console=app.screen
            controller=console._ensure_console_chat_controller()
            runtime=console._console_runtime()
            assert runtime._agent_bridge is not None
            actual_runs=runtime._agent_runs_db
            assert actual_runs is not None and actual_runs.db_path==stores.run_path
            assert actual_runs._execution_owner is runtime._execution_owner
            actual_runs._require_execution_owner()
            await pilot.pause()
            assert not gateway.generations and not tool_dispatches
            assert historical(actual_runs)==original_rows
            report['readiness_resolutions_before_send']=list(gateway.resolutions)
            import traceback
            from dataclasses import asdict
            submit_results=[]
            effect_failures=[]
            notifications=[]
            report.update(submit_results=submit_results,effect_failures=effect_failures,
                          notifications=notifications)
            native_submit=controller.submit_draft
            native_effect=controller._run_durable_postcommit_effect
            native_notify=app.notify
            async def observe_submit(*args,**kwargs):
                result=await native_submit(*args,**kwargs)
                submit_results.append({'accepted':result.accepted,'visible_copy':result.visible_copy,
                    'terminal_status':result.terminal_status.value if result.terminal_status else None,
                    'provider_started':result.provider_started,'session_id':result.session_id})
                return result
            async def observe_effect(preparation_id,effect_name,callback,**kwargs):
                try:
                    return await native_effect(preparation_id,effect_name,callback,**kwargs)
                except BaseException as error:
                    if not effect_failures:
                        effect_failures.append({'effect':effect_name,'type':type(error).__name__,
                            'frames':[{'file':frame.filename,'line':frame.lineno,'name':frame.name}
                                      for frame in traceback.extract_tb(error.__traceback__)]})
                    raise
            def observe_notify(message,*args,**kwargs):
                notifications.append({'message':str(message),'severity':kwargs.get('severity','information')})
                return native_notify(message,*args,**kwargs)
            controller.submit_draft=observe_submit
            controller._run_durable_postcommit_effect=observe_effect
            app.notify=observe_notify
            previous_session_id=controller.store.active_session_id
            console.query_one('#console-new-chat-tab',Button).press()
            await until(lambda:controller.store.active_session_id!=previous_session_id
                        and console._console_visible_draft_session_id==controller.store.active_session_id,pilot,30)
            session_id=controller.store.active_session_id
            session=next(row for row in controller.store.sessions() if row.id==session_id)
            composer=console.query_one('#console-native-composer',ConsoleComposerBar)
            composer.load_draft('Make one new manual response.')
            send_button=console.query_one('#console-send-message',Button)
            try:
                await until(lambda:not send_button.disabled,pilot,30)
            finally:
                report['send_ui']={'active_session':controller.store.active_session_id,
                    'visible_session':console._console_visible_draft_session_id,
                    'disabled':send_button.disabled,'tooltip':str(send_button.tooltip),
                    'provider':session.settings.provider,'model':session.settings.model,
                    'draft_matches':composer.draft_text()=='Make one new manual response.'}
            assert report['send_ui']['draft_matches']
            assert session.settings.provider=='llama_cpp' and session.settings.model=='local-model'
            send_button.press()
            try:
                await until(lambda:(len(gateway.generations)==1
                                    and controller.in_flight_run_count()==0)
                            or bool(effect_failures)
                            or any(result['terminal_status'] in {'blocked','failed','stopped'}
                                   for result in submit_results),pilot)
            finally:
                state=controller.run_state_for(session_id)
                report['send_outcome']={'status':state.status.value,'visible_copy':state.visible_copy,
                    'in_flight_count':controller.in_flight_run_count(),
                    'activity':asdict(controller.activity_for(session_id)),
                    'resolutions':list(gateway.resolutions)}
            assert len(gateway.generations)==1 and controller.in_flight_run_count()==0,report['send_outcome']
            await until(lambda:any(m.content=='Exact fresh manual response Ω'
                                  for m in controller.store.messages_for_session(session_id)),pilot)
            messages=controller.store.messages_for_session(session_id)
            reply=next(m for m in messages if m.content=='Exact fresh manual response Ω')
            saved=next(s for s in controller.store.sessions() if s.id==session_id)
            assert saved.persisted_conversation_id
            report.update(manual_conversation=saved.persisted_conversation_id,manual_message=reply.id,
                          generations=list(gateway.generations),resolutions=list(gateway.resolutions))
            assert historical(actual_runs)==original_rows
    finally:
        # Real app-owned drains; run_test itself invokes native _shutdown.
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        await app.tts_service.wait_closed()
        report['runtime_disposed']=runtime is not None and runtime._disposed
        report['gateway_closed']=gateway.closed
    assert report['runtime_disposed'] and report['gateway_closed']
    assert gateway.generations==[('llama_cpp','local-model')]
    assert not tool_dispatches

try:
    asyncio.run(main())
    with _ordinary_imported_owner(stores,'db.agent_runs') as reopened:
        assert historical(reopened)==original_rows
    chat=CharactersRAGDB(stores.chat_path,'ordinary-manual-reopen')
    try:
        assert chat.get_message_by_id(report['manual_message'])['content']=='Exact fresh manual response Ω'
        assert chat.get_conversation_by_id(report['manual_conversation']) is not None
    finally:chat.close_connection()
    assert require_mobile_cutover(source,stores) is None
    assert claim_facts(operation)==previous['claim']
    report['ordinary_reopen']=True
    assert gateway.generations==[('llama_cpp','local-model')] and not tool_dispatches
finally:
    report.update(source_before=before,source_after=fingerprints(source),
                  generations_after_teardown=list(gateway.generations),tool_dispatches=list(tool_dispatches),
                  resolutions_after_teardown=list(gateway.resolutions),
                  agent_admission_failures=agent_admission_failures,
                  profile_guard=take_violations(),network_guard=blocked_attempts(),
                  exit_home={k:os.environ.get(k) for k in ENTRY_HOME})
    report_path.write_text(json.dumps(report,indent=2))
assert report['source_before']==report['source_after'] and report['exit_home']==ENTRY_HOME
assert protected_roots()==GUARD_ROOTS
assert not report['profile_guard'] and not report['network_guard']
"""


def _child_source(body):
    # Final child facts also survive a body assertion/constructor failure. This
    # footer issues no application operation and cannot turn a failure green.
    footer = r"""
finally:
    exit_record = {
        'entry_home': ENTRY_HOME,
        'exit_home': {key: os.environ.get(key) for key in ENTRY_HOME},
        'profile_guard': take_violations(),
        'network_guard': blocked_attempts(),
        'target_files': {
            str(path): {
                'device': path.stat().st_dev,
                'inode': path.stat().st_ino,
                'mode': path.stat().st_mode,
                'size': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            } if path.is_file() else None
            for path in stores.paths
        },
    }
    report_path.with_name(report_path.stem + '-exit.json').write_text(json.dumps(exit_record, indent=2))
    assert exit_record['entry_home'] == exit_record['exit_home']
    assert not exit_record['profile_guard'] and not exit_record['network_guard']
"""
    return _COMMON + "\ntry:\n" + textwrap.indent(body, "    ") + footer


def run_standalone(result_path):
    entry = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
    uid_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert entry == {"HOME": str(uid_home), "USERPROFILE": None}
    from Tests.network_guard import blocked_attempts
    from Tests.network_guard import install as install_network_guard
    from Tests.real_profile_guard import install as install_profile_guard
    from Tests.real_profile_guard import protected_roots, take_violations

    install_profile_guard()
    install_network_guard()
    assert Path(os.environ["TLDW_TEST_REAL_HOME"]) == uid_home
    assert str(uid_home / ".config/tldw_cli").lower() in protected_roots()
    scratch = Path(os.environ["TMPDIR"]).resolve(strict=True)
    fixture = Path(
        tempfile.mkdtemp(prefix="fresh-mounted-mobile-", dir=scratch)
    ).resolve(strict=True)
    assert fixture.parent == scratch and not fixture.is_relative_to(uid_home)
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join(
            (str(repository), str(repository / "packages/tldw_profile_core/src"))
        ),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    assert {key: environment.get(key) for key in entry} == entry
    report = {
        "fixture": str(fixture),
        "entry_home": entry,
        "children": [],
        "scope": "standalone actual mounted App/scripted provider",
    }
    seed_digest = None
    try:
        for name, body in (("seed", _SEED), ("producer", _PRODUCE), ("app", _APP)):
            if name == "producer":
                assert isinstance(seed_digest, str)
                _read_seed_record(fixture, seed_digest)
            output = fixture / f"native-{name}.json"
            log = fixture / f"native-{name}.log"
            argv = [
                sys.executable,
                "-X",
                "faulthandler",
                "-c",
                _child_source(body),
                str(fixture),
                str(output),
            ]
            if name == "producer":
                argv.append(seed_digest)
            with log.open("w") as stream:
                child = subprocess.Popen(
                    argv,
                    cwd=repository,
                    env=environment,
                    stdout=stream,
                    stderr=stream,
                    start_new_session=True,
                )
                try:
                    code = child.wait(timeout=300)
                except BaseException:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                    report["children"].append(
                        {
                            "name": name,
                            "pid": child.pid,
                            "returncode": child.returncode,
                            "interrupted": True,
                            "log": str(log),
                            "output": str(output),
                        }
                    )
                    raise
            report["children"].append(
                {
                    "name": name,
                    "pid": child.pid,
                    "returncode": code,
                    "log": str(log),
                    "output": str(output),
                }
            )
            result_path.write_text(json.dumps(report, indent=2))
            assert code == 0, log.read_text()[-16000:]
            if name == "seed":
                seed_digest = _read_seed_record(fixture)[1]
        report["completed"] = True
    finally:
        report.update(
            profile_guard=take_violations(),
            network_guard=blocked_attempts(),
            exit_home={key: os.environ.get(key) for key in entry},
        )
        result_path.write_text(json.dumps(report, indent=2))
    assert (
        report["exit_home"] == entry
        and not report["profile_guard"]
        and not report["network_guard"]
    )


if __name__ == "__main__":
    run_standalone(Path(sys.argv[1]))
