"""Actual provider effects require a deliberate recovered OpenAI connection."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SETUP = r"""
import os,sys,json,asyncio,threading,time
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
selector=Path(os.environ['TLDW_CONFIG_PATH']);base=selector.parent.parent;data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n[api_settings.openai]\napi_key="owned-test-key"\napi_base_url="https://api.openai.com/v1"\n[openai_api]\napi_key="owned-test-key"\napi_base_url="https://api.openai.com/v1"\n[providers]\nOpenAI=["gpt-4o-mini"]\n')
selector.chmod(0o600)
from tldw_chatbook.LLM_Calls import LLM_API_Calls as calls
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore,bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,register_pending
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup:startup.close()
authority=admission_authority(root);authority.register('profile',(selector.parent,data))
control=base/'operation';control.mkdir(mode=0o700)
activation=ActivationStore(control/'activation')
def restored(generation='generation'):
 register_pending(root,'restore',('profile',),control,(selector,))
 with authority.maintenance(('profile',),3) as session:
  bind_activation(root,'restore',selector,generation,('config',),session=session)
 (root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
 assert not activation.allowed(generation,'config')
effects=[]
class Response:
 status_code=200
 text=''
 def __bool__(self):return True
 def raise_for_status(self):pass
 def json(self):return {'choices':[{'message':{'content':'answer'}}]}
 def iter_lines(self,**kwargs):yield 'data: {"choices":[{"delta":{"content":"answer"}}]}'
 def iter_content(self,**kwargs):yield b'data: [DONE]\n\n'
 def close(self):effects.append('response-close')
class Session:
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def close(self):effects.append('session-close')
 def mount(self,*args):pass
 def post(self,url,**kwargs):effects.append(('post',url,kwargs));return Response()
calls.requests.Session=Session
message=[{'role':'user','content':'hello'}]
"""

_DENIED = (
    _SETUP
    + r"""
route=sys.argv[1]
if route=='lazy':result=calls.chat_with_openai(message,model='gpt-4o-mini',streaming=True)
restored()
try:
 if route=='lazy':next(result)
 else:calls.chat_with_openai(message,model='gpt-4o-mini',streaming=False)
except Exception:pass
assert not effects,'unreviewed provider effects: '+repr(effects)
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["denied", "lazy"])
def test_unreviewed_openai_never_enters_native_request(tmp_path, route):
    _run(tmp_path, route, "local", script=_DENIED)


_REVIEW = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored()
review=recovery.prepare_openai_reconnect()
assert not effects
assert 'owned-test-key' not in repr(review)
recovery.confirm_openai_reconnect(review)
assert not activation.allowed('generation','config')
result=calls.chat_with_openai(message,model='gpt-4o-mini',streaming=False)
assert result['choices'][0]['message']['content']=='answer'
posts=[e for e in effects if isinstance(e,tuple)]
assert len(posts)==1 and posts[0][1]=='https://api.openai.com/v1/chat/completions'
assert posts[0][2]['headers']['Authorization']=='Bearer owned-test-key'
assert posts[0][2]['allow_redirects'] is False
assert not activation.allowed('generation','config')
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_reviewed_openai_uses_actual_handler_and_selected_connection(tmp_path):
    _run(tmp_path, "review", "local", script=_REVIEW)


_REFUSAL = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored()
route=sys.argv[1]
if route=='anthropic':call=lambda:calls.chat_with_anthropic(message,model='claude-3-haiku',api_key='owned-test-key')
elif route=='local':
 from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import chat_with_custom_openai
 call=lambda:chat_with_custom_openai(message,api_key='owned-test-key')
elif route=='summary':
 from tldw_chatbook.LLM_Calls.Summarization_General_Lib import summarize_with_openai
 call=lambda:summarize_with_openai('owned-test-key','text','summarize')
elif route=='catalog':
 from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import discover_openai_compatible_models
 call=lambda:asyncio.run(discover_openai_compatible_models(provider='anthropic',provider_list_key='anthropic',endpoint='https://api.anthropic.com/v1',api_key='owned-test-key'))
else:
 from tldw_chatbook.LLM_Calls.realtime.transport import WsTransport
 call=lambda:asyncio.run(WsTransport().connect('wss://api.openai.com/v1/realtime',{'Authorization':'Bearer owned-test-key'}))
try:call()
except recovery.ProviderReconnectRequired:pass
else:raise AssertionError('restored unqualified provider body ran')
assert not effects and not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "route", ["anthropic", "local", "summary", "catalog", "realtime"]
)
def test_other_restored_provider_effects_refuse(tmp_path, route):
    _run(tmp_path, route, "local", script=_REFUSAL)


_CATALOG = (
    _SETUP
    + r"""
import httpx
from tldw_chatbook.LLM_Calls import recovery_review as recovery
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import LocalLLMProviderCatalogService
from tldw_chatbook.LLM_Provider_Catalog import openai_compatible_model_discovery as discovery
restored()
recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
requests=[]
def respond(request):
 requests.append(request)
 return httpx.Response(200,json={'data':[{'id':'gpt-4o-mini','object':'model'}]})
client_type=httpx.AsyncClient
def native_client(**kwargs):
 assert kwargs.get('trust_env') is False
 return client_type(transport=httpx.MockTransport(respond),**kwargs)
httpx.AsyncClient=native_client
service=LocalLLMProviderCatalogService()
async def run():
 result=await service.discover_models(provider='openai')
 assert result.status=='success' and result.models[0].model_id=='gpt-4o-mini',result
asyncio.run(run())
assert len(requests)==1 and str(requests[0].url)=='https://api.openai.com/v1/models'
assert requests[0].headers['Authorization']=='Bearer owned-test-key'
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_manual_openai_catalog_uses_reviewed_native_route(tmp_path):
    _run(tmp_path, "catalog", "local", script=_CATALOG)


_RECEIPTS = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored()
route=sys.argv[1]
review=recovery.prepare_openai_reconnect()
recovery.confirm_openai_reconnect(review)
folder=activation._generation('generation')
record=next(folder.glob('openai-connection-*'))
if route=='malformed':record.write_text('{}')
elif route=='ambiguous':
 os.environ['OPENAI_API_KEY']='owned-test-key'
 recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect(auth_source='env:OPENAI_API_KEY'))
elif route=='unrelated':
 (folder/'openai-connection-unrelated.json').write_text('invalid')
elif route=='env':
 os.environ['OPENAI_API_KEY']='owned-test-key'
 record.unlink()
 recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect(auth_source='env:OPENAI_API_KEY'))
 os.environ['OPENAI_API_KEY']='changed-test-key'
else:raise AssertionError(route)
original=recovery._selection
resolved=[]
def observe(*args,**kwargs):
 if kwargs.get('resolve'):resolved.append(args[1] if len(args)>1 else None)
 return original(*args,**kwargs)
recovery._selection=observe
try:result=calls.chat_with_openai(message,model='gpt-4o-mini')
except recovery.ProviderReconnectRequired:
 assert route!='unrelated'
else:
 assert route=='unrelated' and result['choices'][0]['message']['content']=='answer'
if route in ('malformed','ambiguous'):assert not resolved,resolved
if route!='unrelated':assert not effects,effects
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["malformed", "ambiguous", "unrelated", "env"])
def test_exact_receipt_selection_and_credential_drift(tmp_path, route):
    _run(tmp_path, route, "local", script=_RECEIPTS)


_STREAM = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored()
recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
result=calls.chat_with_openai(message,model='gpt-4o-mini',streaming=True)
first=next(result)
assert 'answer' in first and len([e for e in effects if isinstance(e,tuple)])==1
entered=threading.Event();finished=threading.Event();errors=[]
def capture():
 entered.set()
 try:
  with authority.maintenance(('profile',),5):finished.set()
 except BaseException as error:errors.append(error)
thread=threading.Thread(target=capture);thread.start();assert entered.wait(2)
time.sleep(.15);assert not finished.is_set()
if sys.argv[1]=='partial':result.close()
else:assert '[DONE]' in ''.join(result)
thread.join(7)
assert not thread.is_alive() and not errors and finished.is_set(),errors
assert 'response-close' in effects and 'session-close' in effects
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["partial", "exhaust"])
def test_actual_openai_stream_holds_source_until_native_cleanup(tmp_path, route):
    _run(tmp_path, route, "local", script=_STREAM)


_HOSTED = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls.hosted_chat import owned_json_post,HostedHTTPTransportConfig
from tldw_chatbook.Backup_Recovery.control_records import bind_profile
bind_profile(root,selector,('profile',),root/'admission')
result=owned_json_post(config=HostedHTTPTransportConfig(provider='moonshot',base_url='https://api.moonshot.ai/v1',api_key='test-owned',timeout=1,retries=0,retry_delay=0),route='chat/completions',payload={},streaming=True)
entered=threading.Event();finished=threading.Event();errors=[]
def capture():
 entered.set()
 try:
  with authority.maintenance(('profile',),5):finished.set()
 except BaseException as error:errors.append(error)
thread=threading.Thread(target=capture);thread.start();assert entered.wait(2)
time.sleep(.15)
assert not finished.is_set(),'returned native iterator released its source'
result.close();thread.join(7)
assert finished.is_set() and not errors,errors
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_ordinary_owned_provider_iterator_retains_accepted_source(tmp_path):
    _run(tmp_path, "hosted", "local", script=_HOSTED)


_CATALOG_CLOSE = _CATALOG.replace(
    "async def run():\n",
    """async def run():
 original_exit=client_type.__aexit__
 async def broken_close(self,*args):
  await original_exit(self,*args)
  raise RuntimeError('native-close-failure')
 client_type.__aexit__=broken_close
""",
).replace(
    " result=await service.discover_models(provider='openai')\n assert result.status=='success' and result.models[0].model_id=='gpt-4o-mini',result",
    """ try:await service.discover_models(provider='openai')
 except RuntimeError:pass
 else:raise AssertionError('close failure disappeared')
 try:
  with authority.maintenance(('profile',),.2):pass
 except Exception:pass
 else:raise AssertionError('failed async native close released source')
""",
)


def test_catalog_failed_native_close_retains_source(tmp_path):
    _run(tmp_path, "catalog-close", "local", script=_CATALOG_CLOSE)


_ISOLATED_CHILD = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,_launch_descriptor
profile,control,mode=sys.argv[1:];control=Path(control)
select_profile(profile,control)
entry=_launch_descriptor(profile,control)
os.environ['OPENAI_API_KEY']='owned-test-key'  # Fresh deliberate environment setup after profile selection.
from tldw_chatbook.LLM_Calls import LLM_API_Calls as calls,recovery_review as recovery
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
from tldw_chatbook import config
assert config.CLI_APP_CLIENT_ID==entry.installation_id
_,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==entry.config)
activation=ActivationStore(Path(witness['store_root']))
def unrelated_inactive():
 for owner in ('config','skills','mcp.local','runtime.sync_state','db.scheduled_tasks'):
  assert owner in witness['owners'],owner
  assert not activation.allowed(witness['generation'],owner),owner
unrelated_inactive()
if mode=='approve':
 recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect(auth_source='env:OPENAI_API_KEY'))
EFFECT_FIXTURE
answer=calls.chat_with_openai([{'role':'user','content':'hello'}],model='gpt-4o-mini')
assert answer['choices'][0]['message']['content']=='answer'
assert len([e for e in effects if isinstance(e,tuple)])==1
unrelated_inactive()
assert not blocked_attempts()
print('verified isolated reconnect')
""".replace("EFFECT_FIXTURE", _SETUP[_SETUP.index("effects=[]") :])

_ISOLATED = (
    _SETUP
    + r"""
import subprocess
from threading import Event
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
source=base/'archive-source';source.mkdir(mode=0o700)
def config_manifest(doc):
 doc['owners'][0]['owner_id']='config'
 doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
 doc['dependency_groups'][0]['members']=['profile:profile:config']
payload=b'[general]\nusers_name="original"\n[api_settings.openai]\napi_key_env_var="OPENAI_API_KEY"\napi_base_url="https://api.openai.com/v1"\n'.replace(b'\\n',b'\n')
archive=sealed(source,mutate=config_manifest,data=payload)
dest=base/'destinations';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
control=base/'isolated-control'
profile=restore_isolated(archive,plan,control,Event())
env=os.environ.copy();env['OPENAI_API_KEY']='owned-test-key'
for mode in ('approve','reopen'):
 result=subprocess.run([sys.executable,'-c',CHILD,profile,str(control),mode],env=env,capture_output=True,text=True,timeout=20)
 assert result.returncode==0,result.stderr[-6000:]
 assert 'verified isolated reconnect' in result.stdout
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_isolated_generation_reopens_openai_receipt_in_fresh_process(tmp_path):
    _run(
        tmp_path,
        "isolated",
        "local",
        script="CHILD=" + repr(_ISOLATED_CHILD) + "\n" + _ISOLATED,
        timeout=60,
    )


_DAMAGED_CHILD = (
    _ISOLATED_CHILD.split("if mode=='approve':")[0]
    + r"""
import json
stage,damage=mode.split(':')
EFFECT_FIXTURE
review=None
if stage!='prepare':
 review=recovery.prepare_openai_reconnect(auth_source='env:OPENAI_API_KEY')
if stage=='request':
 recovery.confirm_openai_reconnect(review)
 operation=recovery._Operation(Path(entry.config))
generation=activation._generation(witness['generation'])
before={p.name:p.read_bytes() for p in generation.glob('openai-connection-*')}
required=generation/'required.json'
association=bootstrap.default_bootstrap_root()/('activation-'+bootstrap._key(entry.config)+'.json')
if damage=='missing_required':required.unlink()
elif damage=='corrupt_required':required.write_bytes(b'{')
elif damage=='missing_pair':association.unlink()
elif damage=='mismatched_pair':
 document=json.loads(association.read_bytes())
 document['activation']['generation']='foreign-generation'
 association.write_text(json.dumps(document))
else:raise AssertionError(damage)
select=recovery._selection
def no_secret(*args,**kwargs):
 if kwargs.get('resolve'):raise AssertionError('damaged history resolved credentials')
 return select(*args,**kwargs)
recovery._selection=no_secret
def no_client(*args,**kwargs):raise AssertionError('damaged history constructed client')
calls.requests.Session=no_client
try:
 if stage=='prepare':recovery.prepare_openai_reconnect(auth_source='env:OPENAI_API_KEY')
 elif stage=='confirm':recovery.confirm_openai_reconnect(review)
 else:
  with recovery._using(operation):
   recovery.openai_post(calls.requests.Session(),'https://api.openai.com/v1/chat/completions',headers={'Authorization':'Bearer owned-test-key'})
except (recovery.ProviderReconnectRequired,bootstrap.RecoveryRequired):pass
else:raise AssertionError('damaged paired history authorized provider')
finally:
 if stage=='request':operation.close()
assert {p.name:p.read_bytes() for p in generation.glob('openai-connection-*')}==before
assert not effects and not blocked_attempts()
print('verified isolated reconnect')
""".replace("EFFECT_FIXTURE", _SETUP[_SETUP.index("effects=[]") :])
)


@pytest.mark.parametrize("stage", ["prepare", "confirm", "request"])
@pytest.mark.parametrize(
    "damage",
    ["missing_required", "corrupt_required", "missing_pair", "mismatched_pair"],
)
def test_actual_isolated_paired_damage_refuses_before_provider_effects(
    tmp_path, stage, damage
):
    script = _ISOLATED.replace(
        "for mode in ('approve','reopen'):", "for mode in (sys.argv[1],):"
    )
    _run(
        tmp_path,
        f"{stage}:{damage}",
        "local",
        script="CHILD=" + repr(_DAMAGED_CHILD) + "\n" + script,
        timeout=60,
    )


_CONSOLE = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway,ConsoleProviderResolution
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
entered=threading.Event();release=threading.Event();settled=threading.Event();errors=[]
original_post=Session.post
def blocked(self,*args,**kwargs):
 entered.set();assert release.wait(8)
 return original_post(self,*args,**kwargs)
Session.post=blocked
original_close=Response.close
def closed(self):original_close(self);settled.set()
Response.close=closed
async def run():
 gateway=ConsoleProviderGateway()
 resolution=ConsoleProviderResolution(provider='OpenAI',execution_key='openai',readiness_key='openai',base_url='https://api.openai.com/v1',model='gpt-4o-mini',ready=True,api_key='owned-test-key')
 from tldw_chatbook.Chat.console_prepared_request import build_console_request,prepare_provider_request,resolve_request_capacity
 request=prepare_provider_request(build_console_request(message),wire_style='single_preamble',model=resolution.model,provider=resolution.provider,capacity=resolve_request_capacity(context_window_tokens=None),count_fn=lambda messages,model:10)
 stream=gateway._stream_generic_chat(resolution,request)
 task=asyncio.create_task(anext(stream))
 assert await asyncio.to_thread(entered.wait,3)
 task.cancel()
 # A separate thread releases native I/O even if close wrongly blocks this loop.
 timer=threading.Timer(1,release.set);timer.start()
 start=time.monotonic()
 try:await task
 except asyncio.CancelledError:pass
 assert time.monotonic()-start<.5,'cancelled waiter blocked on native close'
 assert not settled.is_set()
 try:
  with authority.maintenance(('profile',),.15):pass
 except Exception:pass
 else:raise AssertionError('cancelled native provider released its source')
 release.set();assert await asyncio.to_thread(settled.wait,3)
 await gateway.aclose();timer.join(2)
asyncio.run(run())
with authority.maintenance(('profile',),3):pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_cancelled_actual_console_waiter_detaches_but_native_source_remains(tmp_path):
    _run(tmp_path, "console", "local", script=_CONSOLE)


_RESPONSES = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
def lines(self,**kwargs):
 yield 'data: {"type":"response.output_text.delta","delta":"answer"}'
 yield 'data: {"type":"response.completed","response":{}}'
Response.iter_lines=lines
Response.json=lambda self:{'output_text':'answer','id':'response-local'}
route=sys.argv[1]
result=calls.chat_with_openai(message,model='gpt-5-mini',reasoning_effort='low',streaming=route!='complete')
if route=='complete':assert result['choices'][0]['message']['content']=='answer'
else:
 assert 'answer' in next(result)
 if route=='partial':result.close()
 else:assert '[DONE]' in ''.join(result)
posts=[e for e in effects if isinstance(e,tuple)]
assert len(posts)==1 and posts[0][1]=='https://api.openai.com/v1/responses'
assert 'response-close' in effects and 'session-close' in effects
with authority.maintenance(('profile',),3):pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["partial", "exhaust", "complete"])
def test_reviewed_actual_responses_modes_close_cleanly(tmp_path, route):
    _run(tmp_path, route, "local", script=_RESPONSES)


_AUTOREFRESH = _CATALOG.replace(
    "async def run():\n",
    """async def run():
 from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import ModelCatalogSettings
 try:await service.refresh_stale_configured_providers(catalog_settings=ModelCatalogSettings(refresh_consent_recorded=True),disk_store=None)
 except recovery.ProviderReconnectRequired:pass
 else:raise AssertionError('imported auto-refresh consent replayed')
 assert not requests
""",
)


def test_imported_catalog_auto_refresh_consent_stays_inactive(tmp_path):
    _run(tmp_path, "automatic", "local", script=_AUTOREFRESH)


_AUTH_SOURCE = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored()
os.environ['OPENAI_API_KEY']='owned-test-key'
get=os.environ.get;reads=[];allow=False
def selected_get(key,*args):
 if key=='OPENAI_API_KEY':
  assert allow,'preview resolved credential'
  reads.append(key)
 if key in ('ANTHROPIC_API_KEY','OPENROUTER_API_KEY'):raise AssertionError('unrelated credential resolved')
 return get(key,*args)
os.environ.get=selected_get
review=recovery.prepare_openai_reconnect(auth_source='env:OPENAI_API_KEY')
assert not reads and not effects
allow=True
recovery.confirm_openai_reconnect(review)
def global_bridge():raise AssertionError('global credential bridge invoked')
calls.load_settings=global_bridge
answer=calls.chat_with_openai(message,model='gpt-4o-mini')
assert answer['choices'][0]['message']['content']=='answer'
assert reads and all(key=='OPENAI_API_KEY' for key in reads)
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_preview_does_not_resolve_and_execution_reads_only_selected_auth_source(
    tmp_path,
):
    _run(tmp_path, "auth-source", "local", script=_AUTH_SOURCE)


_RETRY_REDIRECT = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
route=sys.argv[1];count=0
original=Session.post
class Rejected(Response):
 status_code=400 if route=='retry' else 302
 text='unsupported stream_options'
def post(self,*args,**kwargs):
 global count
 count+=1
 response=original(self,*args,**kwargs)
 return Rejected() if count==1 else response
Session.post=post
try:
 result=calls.chat_with_openai(message,model='gpt-4o-mini',streaming=True,api_base_url='https://elsewhere.invalid/v1' if route=='override' else None)
 chunks=list(result)
except Exception:
 assert route in ('redirect','override')
else:
 if route=='retry':assert 'answer' in ''.join(chunks)
 else:assert 'provider_reconnect_required' in ''.join(chunks) and 'answer' not in ''.join(chunks)
posts=[e for e in effects if isinstance(e,tuple)]
assert len(posts)==({'retry':2,'redirect':1,'override':0}[route])
assert all(e[2]['allow_redirects'] is False for e in posts)
if route=='retry':assert 'stream_options' not in posts[1][2]['json']
with authority.maintenance(('profile',),3):pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["retry", "redirect", "override"])
def test_actual_openai_retry_keeps_reviewed_endpoint_and_refuses_redirect(
    tmp_path, route
):
    _run(tmp_path, route, "local", script=_RETRY_REDIRECT)


_LEGACY = _REVIEW
_LEGACY = _LEGACY.replace(
    'api_key="owned-test-key"\\napi_base_url', 'api_key=""\\napi_base_url'
).replace(
    "[providers]\\nOpenAI=",
    '[API]\\nopenai_api_key="owned-test-key"\\n[providers]\\nOpenAI=',
)


def test_actual_legacy_openai_config_source_can_be_reviewed(tmp_path):
    _run(tmp_path, "legacy", "local", script=_LEGACY)


_CLOSE_FAILED = (
    _SETUP
    + r"""
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
result=calls.chat_with_openai(message,model='gpt-4o-mini',streaming=True)
assert 'answer' in next(result)
original=Response.close
Response.close=lambda self:(_ for _ in ()).throw(RuntimeError('native close failed'))
try:result.close()
except RuntimeError:pass
else:raise AssertionError('native close failure disappeared')
try:
 with authority.maintenance(('profile',),.2):pass
except Exception:pass
else:raise AssertionError('failed native close released source')
Response.close=original;result.close()
with authority.maintenance(('profile',),3):pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_openai_close_failure_holds_until_successful_explicit_retry(tmp_path):
    _run(tmp_path, "failed-close", "local", script=_CLOSE_FAILED)


_CATALOG_CANCEL = (
    _SETUP
    + r"""
import httpx
from tldw_chatbook.LLM_Calls import recovery_review as recovery
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import LocalLLMProviderCatalogService
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
client_type=httpx.AsyncClient
async def run():
 entered=asyncio.Event();closing=asyncio.Event();release=asyncio.Event();closed=asyncio.Event()
 class Body(httpx.AsyncByteStream):
  async def __aiter__(self):
   entered.set();await asyncio.Event().wait();yield b'{}'
  async def aclose(self):
   closing.set();await release.wait();closed.set()
 def respond(request):return httpx.Response(200,stream=Body())
 httpx.AsyncClient=lambda **kwargs:client_type(transport=httpx.MockTransport(respond),**kwargs)
 task=asyncio.create_task(LocalLLMProviderCatalogService().discover_models(provider='openai'))
 await asyncio.wait_for(entered.wait(),3);task.cancel()
 await asyncio.wait_for(closing.wait(),3)
 assert not task.done() and not closed.is_set()
 try:
  with authority.maintenance(('profile',),.15):pass
 except Exception:pass
 else:raise AssertionError('cancelled async cleanup released native source')
 release.set()
 try:await task
 except asyncio.CancelledError:pass
 assert closed.is_set()
asyncio.run(run())
with authority.maintenance(('profile',),3):pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_cancelled_catalog_holds_until_actual_async_response_close(tmp_path):
    _run(tmp_path, "cancel", "local", script=_CATALOG_CANCEL)


_GENERATION = (
    _SETUP
    + r"""
import shutil
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
old=next(activation._generation('generation').glob('openai-connection-*'))
control2=base/'operation2';control2.mkdir(mode=0o700)
register_pending(root,'restore2',('profile',),control2,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore2',selector,'generation2',('config',),session=session)
(root/('pending-'+bootstrap._key('restore2')+'.json')).unlink()
new=ActivationStore(control2/'activation');assert not new.allowed('generation2','config')
shutil.copyfile(old,new._generation('generation2')/old.name)
(new._generation('generation2')/old.name).chmod(0o600)
try:calls.chat_with_openai(message,model='gpt-4o-mini')
except recovery.ProviderReconnectRequired:pass
else:raise AssertionError('historical generation receipt was replayed')
assert not effects and not blocked_attempts()
print('retired and reopened')
"""
)


def test_imported_openai_receipt_cannot_authorize_a_new_generation(tmp_path):
    _run(tmp_path, "generation", "local", script=_GENERATION)


_ORDINARY = (
    _SETUP
    + r"""
from tldw_chatbook.Backup_Recovery.control_records import bind_profile
bind_profile(root,selector,('profile',),root/'admission')
streaming=sys.argv[1]=='stream'
result=calls.chat_with_openai(message,model='gpt-4o-mini',streaming=streaming)
if streaming:assert 'answer' in ''.join(result)
else:assert result['choices'][0]['message']['content']=='answer'
posts=[e for e in effects if isinstance(e,tuple)]
assert len(posts)==1 and 'allow_redirects' not in posts[0][2]
assert not blocked_attempts()
with authority.maintenance(('profile',),3):pass
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["stream", "complete"])
def test_enrolled_ordinary_openai_handler_preserves_transport_behavior(tmp_path, route):
    _run(tmp_path, route, "local", script=_ORDINARY)


_NETRC = (
    _SETUP
    + r"""
import requests
from tldw_chatbook.LLM_Calls import recovery_review as recovery
restored();recovery.confirm_openai_reconnect(recovery.prepare_openai_reconnect())
prepared=[]
class OfflineAdapter(requests.adapters.BaseAdapter):
 def send(self,request,**kwargs):
  prepared.append(request)
  response=requests.Response();response.status_code=200;response._content=b'{"choices":[{"message":{"content":"answer"}}]}'
  response._content_consumed=True;response.request=request
  return response
 def close(self):pass
class NativeSession(requests.sessions.Session):
 def mount(self,prefix,adapter):super().mount(prefix,OfflineAdapter())
def unreviewed_netrc(*args,**kwargs):raise AssertionError('ambient netrc auth resolution')
requests.sessions.get_netrc_auth=unreviewed_netrc
requests.Session=NativeSession
result=calls.chat_with_openai(message,model='gpt-4o-mini')
assert result['choices'][0]['message']['content']=='answer'
assert len(prepared)==1 and prepared[0].headers['Authorization']=='Bearer owned-test-key'
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_requests_preparation_never_resolves_unreviewed_netrc_auth(tmp_path):
    _run(tmp_path, "netrc", "local", script=_NETRC)


_DESCRIPTOR = (
    _SETUP
    + r"""
import httpx
from tldw_chatbook.Backup_Recovery.control_records import bind_profile
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.LLM_Calls.recovery_review import ProviderReconnectRequired
route=sys.argv[1]
if route=='recovered':restored()
else:bind_profile(root,selector,('profile',),root/'admission')
async def run():
 entered=asyncio.Event();release=asyncio.Event();closed=asyncio.Event();requests=[]
 class Body(httpx.AsyncByteStream):
  async def __aiter__(self):yield b'{"answer":"local"}'
  async def aclose(self):
   entered.set();await release.wait();closed.set()
 async def handle(request):
  requests.append(request)
  return httpx.Response(200,stream=Body())
 async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
  gateway=ConsoleProviderGateway(http_client=client)
  if route=='recovered':
   try:await gateway._post_without_high_level_http_log(client,'http://localhost/v1/chat/completions',json_payload={'message':'test'})
   except ProviderReconnectRequired:pass
   else:raise AssertionError('restored native local POST was admitted')
   assert not requests
  else:
   task=asyncio.create_task(gateway._post_without_high_level_http_log(client,'http://localhost/v1/chat/completions',json_payload={'message':'test'}))
   await asyncio.wait_for(entered.wait(),3)
   assert len(requests)==1 and not task.done() and not closed.is_set()
   try:
    with authority.maintenance(('profile',),.15):pass
   except Exception:pass
   else:raise AssertionError('awaited native response close released source')
   release.set();response=await task
   assert response.json()=={'answer':'local'} and closed.is_set()
asyncio.run(run())
with authority.maintenance(('profile',),3):pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["ordinary", "recovered"])
def test_actual_console_static_post_instance_call_keeps_awaited_native_scope(
    tmp_path, route
):
    _run(tmp_path, route, "local", script=_DESCRIPTOR)
