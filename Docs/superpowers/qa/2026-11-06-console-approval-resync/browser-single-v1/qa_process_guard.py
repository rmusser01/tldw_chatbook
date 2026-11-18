"""Private selected-source process protection, including unchanged -I workers.

Plain-path .pth selects this module and product root BEFORE shared dependencies.
No product imports here. The original real-profile guard installs BEFORE HOME
redirects. Only individually registered owned127.0.0.1 ports and the exact
stdlib Windows asyncio self-pipe call chain may connect. No API fake/patch.
"""
from __future__ import annotations
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

RUN=Path(__file__).resolve().parent
SELECTION=json.loads((RUN/'source-selection.json').read_text(encoding='utf-8-sig'))
REPO=Path(SELECTION['source_root']).resolve()
CORE=REPO/'packages/tldw_profile_core/src'
PRODUCT_PREFIXES=('tldw_chatbook','tldw_profile_core')
PROFILE=(RUN/'product-profile').resolve()
sys.dont_write_bytecode=True
if any(name.startswith(PRODUCT_PREFIXES) for name in sys.modules):raise RuntimeError('QA_product_import_before_source_guard')
sys.path.insert(0,str(REPO));sys.path.insert(1,str(CORE))
if not Path(importlib.util.find_spec('Tests').origin).resolve().is_relative_to(REPO):raise RuntimeError('QA_wrong_Test_guard_origin')
if not Path(importlib.util.find_spec('tldw_chatbook').origin).resolve().is_relative_to(REPO):raise RuntimeError('QA_wrong_product_package_origin')
if not Path(importlib.util.find_spec('tldw_profile_core').origin).resolve().is_relative_to(CORE):raise RuntimeError('QA_wrong_profile_core_package_origin')
real_home=Path(SELECTION['original_home']).resolve()
if real_home==PROFILE or real_home.is_relative_to(RUN):raise RuntimeError('QA_original_home_cannot_be_private_profile')
os.environ['TLDW_TEST_REAL_HOME']=str(real_home)
from Tests import real_profile_guard
real_profile_guard.install()  # MUST precede redirect and any product import.
ORIGINAL_HOME=str(real_profile_guard._real_home())
if Path(ORIGINAL_HOME).resolve()!=real_home:raise RuntimeError('QA_original_home_guard_anchor_mismatch')
# Start with Windows execution essentials, then add ONLY owned fixture/profile
# and genuine stock served-child transport variables. No ambient credentials,
# config/provider overrides or proxies can reach the product/stub. Values are
# never copied into receipts. Real-profile guard is already installed above.
WINDOWS_ENV_ALLOWLIST={'SYSTEMROOT','WINDIR','SYSTEMDRIVE','COMSPEC','PATH','PATHEXT','PROCESSOR_ARCHITECTURE','PROCESSOR_ARCHITEW6432','PROCESSOR_IDENTIFIER','NUMBER_OF_PROCESSORS','PROGRAMFILES','PROGRAMFILES(X86)','PROGRAMW6432','COMMONPROGRAMFILES','COMMONPROGRAMFILES(X86)','COMMONPROGRAMW6432','PROGRAMDATA','PUBLIC','LANG','LC_ALL','TERM','COLORTERM','TLDW_TEST_REAL_HOME'}
STOCK_SERVED_ENV={'TEXTUAL_DRIVER','TEXTUAL_FPS','TEXTUAL_COLOR_SYSTEM','TERM_PROGRAM','TERM_PROGRAM_VERSION','COLUMNS','ROWS','CHATBOOK_SERVED_CHILD'}
served_child=os.environ.get('CHATBOOK_SERVED_CHILD')=='1'
allowed=WINDOWS_ENV_ALLOWLIST | (STOCK_SERVED_ENV if served_child else set())
removed=[name for name in os.environ if name.upper() not in allowed]
for name in removed:os.environ.pop(name,None)
ENVIRONMENT_SCRUB={'before_product_import':True,'removed_variable_count':len(removed),'allowlist_names':sorted(allowed),'stock_served_transport_preserved':served_child,'ambient_proxy_credentials_removed':True,'ordinary_CPython_cache_validation':True,'bytecode_writes':False}
for name,relative in {'HOME':'home','USERPROFILE':'home','XDG_CONFIG_HOME':'config','XDG_DATA_HOME':'data','TLDW_CONFIG_PATH':'config/config.toml','TLDW_TEST_CONFIG_ROOT':'.','TEMP':'temp','TMP':'temp','APPDATA':'appdata','LOCALAPPDATA':'localappdata','XDG_CACHE_HOME':'cache'}.items():
    location=PROFILE/relative
    if name!='TLDW_CONFIG_PATH':location.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(location)
os.environ.pop('TLDW_TEST_CONFIG_ROOT_OWNER',None)
os.environ.update(USERNAME='private-qa',USERDOMAIN='PRIVATE-QA',NO_PROXY='127.0.0.1,::1,localhost',TLDW_CANVAS_ENABLED='false',PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
IS_WORKER=sys.argv[0]=='-m'
LOCK=threading.RLock()
PORTS={}
NETWORK={'owned_connects':0,'stdlib_selfpipe_connects':0,'refused_connects':0,'refused_dns':0,'loopback_auth_validations':0}
OBSERVATIONS=[]
EVENTS=RUN/('guard-events-'+str(os.getpid())+'.jsonl')

def record(event,**facts):
    row={'event':event,'pid':os.getpid(),'ppid':os.getppid(),'timestamp_ns':time.perf_counter_ns(),**facts}
    with LOCK,EVENTS.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,default=str)+'\n')

def allow_loopback_port(port,purpose,*,owner_pid):
    """Trusted QA setup only, after own listener/verified child metadata exists.

    Browser parent may register its own served/proxy listeners and exact child
    control/provider ports only after private PID/nonce/metadata verification.
    An isolated physical tool worker is never granted these HTTP connections.
    """
    if IS_WORKER or type(port) is not int or not 0<port<65536 or type(owner_pid) is not int or owner_pid<=0:
        raise RuntimeError('QA_invalid_owned_endpoint')
    if purpose not in {'scripted_provider','native_control','served_site','browser_deny_proxy','child_control','child_provider'}:
        raise ValueError('QA_unknown_endpoint_purpose')
    with LOCK:PORTS[port]={'purpose':purpose,'owner_pid':owner_pid}
    record('owned_endpoint_registered',port=port,purpose=purpose,owner_pid=owner_pid)

def _selfpipe(address):
    if not isinstance(address,tuple) or address[0]!='127.0.0.1':return False
    frame=sys._getframe(2)
    socket_source=Path(sys.base_prefix)/'Lib/socket.py'
    proactor_source=Path(sys.base_prefix)/'Lib/asyncio/proactor_events.py'
    if Path(frame.f_code.co_filename)!=socket_source or frame.f_code.co_name!='_fallback_socketpair':return False
    while frame is not None:
        if Path(frame.f_code.co_filename)==proactor_source and frame.f_code.co_name=='_make_self_pipe':return True
        frame=frame.f_back
    return False

def _stock_loopback_auth_validation(args):
    # Actual stock policy has an early literal-IP return; only localhost uses
    # this resolver. This grants NO connection and changes no resolver/result.
    if IS_WORKER or args[0]!='localhost' or args[1] is not None or args[3]!=1:return False
    frame=sys._getframe(2)
    if Path(frame.f_code.co_filename)==Path(sys.base_prefix)/'Lib/socket.py' and frame.f_code.co_name=='getaddrinfo':frame=frame.f_back
    module=sys.modules.get('tldw_chatbook.Canvas.web_auth')
    original=getattr(module,'is_loopback_host',None)
    return callable(original) and frame.f_code is original.__code__ and Path(frame.f_code.co_filename).resolve()==REPO/'tldw_chatbook/Canvas/web_auth.py'

def audit(event,args):
    if event=='socket.connect':
        address=args[1]
        if _selfpipe(address):
            with LOCK:NETWORK['stdlib_selfpipe_connects']+=1
            record('stdlib_asyncio_selfpipe',address=address)
            return
        with LOCK:entry=PORTS.get(address[1]) if isinstance(address,tuple) and len(address)>=2 and address[0]=='127.0.0.1' else None
        if entry is None:
            with LOCK:NETWORK['refused_connects']+=1
            record('network_refused',kind='connect',loopback_literal=isinstance(address,tuple) and address[0]=='127.0.0.1')
            raise RuntimeError('QA_unowned_network_refused')
        with LOCK:NETWORK['owned_connects']+=1
    elif event=='socket.getaddrinfo':
        if _stock_loopback_auth_validation(args):
            with LOCK:NETWORK['loopback_auth_validations']+=1
            record('stock_loopback_auth_address_validation',source='tldw_chatbook/Canvas/web_auth.py:is_loopback_host',port_is_none=True,stream_type=True)
            return
        with LOCK:allowed=args[0]=='127.0.0.1' and args[1] in PORTS
        if not allowed:
            with LOCK:NETWORK['refused_dns']+=1
            record('network_refused',kind='DNS')
            raise RuntimeError('QA_unowned_DNS_refused')
    elif event=='subprocess.Popen':
        record('physical_process_launch',executable=str(args[0]),argument_count=len(args[1]) if isinstance(args[1],(list,tuple)) else None)
sys.addaudithook(audit)
record('protected_process_started',executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,source_root=str(REPO),original_home=ORIGINAL_HOME,private_profile=str(PROFILE),guard_before_redirection=True,environment_scrub=ENVIRONMENT_SCRUB,isolated_worker=IS_WORKER,isolated_mode=bool(sys.flags.isolated))

def verify_frozen():
    """Run once before product imports in app/parent; worker closure checked later."""
    path=Path(SELECTION['freeze_path']).resolve()
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    if Path(value['source_root']).resolve()!=REPO or value.get('sampling_authorized') is not True:raise RuntimeError('QA_frozen_source_not_authorized')
    revision=subprocess.run(['git','-c','safe.directory='+REPO.as_posix(),'-C',str(REPO),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!=value['revision']:raise RuntimeError('QA_frozen_revision_changed')
    if not {'tldw_chatbook/app.py','packages/tldw_profile_core/src/tldw_profile_core/__init__.py'}.issubset(value['sha256']):raise RuntimeError('QA_frozen_product_hashes_missing')
    for relative,expected in value['sha256'].items():
        source=(REPO/relative).resolve()
        if not source.is_relative_to(REPO) or hashlib.sha256(source.read_bytes()).hexdigest()!=expected.lower():raise RuntimeError('QA_frozen_source_hash_changed:'+relative)
    record('frozen_source_verified',revision=revision,path=str(path),freeze_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),product_hash_count=len(value['sha256']))
    return value

# Content-free physical worker observation only. Function bodies, native calls,
# permissions, handles and return values remain original. Main startup is NOT
# globally profiled. Observer cost remains included in physical worker timing.
TARGETS={'workspace_root_pin.py':{'_windows_open_directory','_windows_set_current_directory','_pin_windows'},'workspace_tool_worker.py':{'run_workspace_worker','_emit'},'workspace_tool_dispatch.py':{'execute_pinned_operation'}}
def physical_profile(frame,event,arg):
    if event not in {'call','return'}:return
    name=frame.f_code.co_name;raw_filename=frame.f_code.co_filename
    if name not in TARGETS.get(os.path.basename(raw_filename),set()):return
    filename=Path(raw_filename)
    if not filename.is_relative_to(REPO):return
    facts={'file':str(filename.relative_to(REPO)),'function':name,'phase':event}
    if name=='_windows_open_directory' and event=='return' and isinstance(arg,tuple):facts.update(native_handle=arg[0],identity=repr(arg[1]),reparse=arg[2])
    if name=='_emit' and event=='call':
        payload=frame.f_locals.get('payload',{})
        facts['frame']={key:payload.get(key) for key in ('operation_id','outcome','code','elapsed_ms','truncated','cleanup_proven')}
        result=payload.get('result')
        if isinstance(result,str):facts.update(result_bytes=len(result.encode('utf-8')),result_sha256=hashlib.sha256(result.encode('utf-8')).hexdigest())
    if name=='execute_pinned_operation' and event=='call':
        request=frame.f_locals.get('request');facts.update(operation=getattr(request,'operation',None),operation_id=getattr(request,'operation_id',None))
    if name=='run_workspace_worker' and event=='return':facts['exit_code']=arg
    record('physical_worker_source',**facts)
if IS_WORKER:sys.setprofile(physical_profile)

def finish():
    sys.setprofile(None)
    frozen=json.loads(Path(SELECTION['freeze_path']).read_text(encoding='utf-8-sig'))
    modules={};mismatches=[]
    for name,module in tuple(sys.modules.items()):
        if not name.startswith(PRODUCT_PREFIXES) and name!='__main__':continue
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix=='.pyc':path=Path(importlib.util.source_from_cache(str(path)))
        if name=='__main__' and not any(path.is_relative_to(base) for base in (REPO/'tldw_chatbook',CORE/'tldw_profile_core')):continue
        within=path.is_relative_to(REPO)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        relative=path.relative_to(REPO).as_posix() if within else None
        expected=frozen['sha256'].get(relative)
        matches=expected is not None and digest==expected.lower()
        modules[name]={'path':str(path),'within_selected_root':within,'sha256':digest,'matches_frozen_source':matches}
        if not within or not matches:mismatches.append(name)
    violations=len(real_profile_guard._violations)
    result={'pid':os.getpid(),'ppid':os.getppid(),'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'source_root':str(REPO),'original_home':ORIGINAL_HOME,'guard_before_redirection':True,'isolated_worker':IS_WORKER,'isolated_mode':bool(sys.flags.isolated),'network':NETWORK,'environment_scrub':ENVIRONMENT_SCRUB,'modules':modules,'source_mismatches':mismatches,'real_profile_refusals':violations,'qualified':False,'product_samples':0}
    (RUN/('process-'+str(os.getpid())+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if mismatches or violations:print('QA_guard_final_evidence_requires_failure_review',file=sys.stderr)
atexit.register(finish)
