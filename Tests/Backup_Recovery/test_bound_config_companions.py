"""An intact exact config binding must allow its installed owner's startup I/O."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import os
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile
home=Path.home();parent=home/'.config'/'tldw_cli';parent.mkdir(parents=True,mode=0o700)
selected=parent/'config.toml';data=home/'data';data.mkdir(mode=0o700)
(data/'fixture').mkdir(mode=0o700)
selected.write_text(f'[general]\nusers_name="fixture"\n[paths]\ndata_dir="{data.as_posix()}"\n')
selected.chmod(0o600);selected.with_name(selected.name+'.lock').touch(mode=0o600)
os.environ['TLDW_CONFIG_PATH']=str(selected)
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
authority.register('profile',(selected,data))
bind_profile(root,selected,('profile',),root/'admission')
before=bootstrap._records(root)[1]
assert str(parent) not in before[0]['roots']
assert bootstrap.startup_permission(selected,root)==(True,'startup_allowed')
from tldw_chatbook import config
assert config.get_cli_setting('general','users_name')=='fixture'
assert bootstrap._records(root)[1]==before
print('retired and reopened')
'''


@pytest.mark.parametrize("explicit_selector", [False, True])
def test_config_owner_reopens_with_exact_binding_and_protected_parent(
    tmp_path, explicit_selector
):
    script = _SCRIPT
    if not explicit_selector:
        script = script.replace(
            "os.environ['TLDW_CONFIG_PATH']=str(selected)",
            "os.environ.pop('TLDW_CONFIG_PATH',None)",
        )
    _run(tmp_path, "bound", "startup", script=script, timeout=30)


def test_bound_config_only_profile_creates_user_directory_inside_owned_root(tmp_path):
    script = _SCRIPT.replace("(data/'fixture').mkdir(mode=0o700)\n", "").replace(
        "from tldw_chatbook import config",
        "registry_before=bootstrap._registry(root)\nfrom tldw_chatbook import config",
    ).replace(
        "print('retired and reopened')",
        "from tldw_chatbook.Utils.platform_files import os as native_os\n"
        "assert config.get_user_data_dir()==data/'fixture'\n"
        "assert native_os.stat(data/'fixture').st_mode & 0o077 == 0\n"
        "assert bootstrap._registry(root)==registry_before\n"
        "print('retired and reopened')",
    )
    _run(tmp_path, "bound", "missing-user-data", script=script, timeout=30)


@pytest.mark.parametrize("explicit_selector", [False, True])
def test_ordinary_unbound_config_parent_behavior_is_unchanged(tmp_path, explicit_selector):
    script = _SCRIPT.replace(
        "bind_profile(root,selected,('profile',),root/'admission')", ""
    ).replace("assert str(parent) not in before[0]['roots']", "assert before==[]")
    if not explicit_selector:
        script = script.replace(
            "os.environ['TLDW_CONFIG_PATH']=str(selected)",
            "os.environ.pop('TLDW_CONFIG_PATH',None)",
        )
    script = script.replace(
        "print('retired and reopened')",
        f"assert config.application_owned_config_directory(selected)=={('None' if explicit_selector else 'parent')}\nprint('retired and reopened')",
    )
    _run(tmp_path, "ordinary", "startup", script=script, timeout=30)


_DEFAULT_DIRECTORIES = _SCRIPT.replace(
    "(data/'fixture').mkdir(mode=0o700)\n", ""
).replace(
    "selected=parent/'config.toml';data=home/'data';data.mkdir(mode=0o700)",
    "selected=parent/'config.toml';data=home/'.local/share/tldw_cli/fixture'\n"
    "for folder in (home/'.local',home/'.local/share',data.parent,data):folder.mkdir(mode=0o700)\n"
    "(data/'chat_dicts').mkdir(mode=0o700)\n"
    "store=data/'existing-owned-file';store.write_bytes(b'original');store.chmod(0o600)",
).replace(
    "[paths]\\ndata_dir=\"{data.as_posix()}\"\\n", ""
).replace(
    "os.environ['TLDW_CONFIG_PATH']=str(selected)", "os.environ.pop('TLDW_CONFIG_PATH',None)"
).replace(
    "authority.register('profile',(selected,data))",
    "authority.register('profile',(selected,store,data/'chat_dicts'))",
).replace(
    "assert config.get_cli_setting('general','users_name')=='fixture'",
    "assert config.get_user_data_dir()==data\n"
    "assert store.read_bytes()==b'original'\n"
    "assert not (home/'.tldw_cli-data-root.lock').exists()",
)


def test_bound_default_data_parent_is_verified_without_creation_authority(tmp_path):
    _run(tmp_path, "bound", "default-data", script=_DEFAULT_DIRECTORIES, timeout=30)


@pytest.mark.parametrize("mode", ["unsafe", "missing", "alias", "external", "ambiguous"])
def test_bound_data_verification_never_repairs_or_selects_other_directories(tmp_path, mode):
    changes = {
        "unsafe": "import sys,subprocess\n"
        "from tldw_chatbook.Utils.platform_files import os as native_os\n"
        "if sys.platform=='win32':\n"
        " subprocess.run(['icacls',str(data),'/grant','*S-1-1-0:(R)'],check=True,capture_output=True)\n"
        "else:data.chmod(0o755)\n"
        "assert native_os.stat(data).st_mode & 0o044",
        "missing": "data.rename(home/'preserved-data')",
        "alias": "data.rename(home/'preserved-data');data.symlink_to(home/'preserved-data',target_is_directory=True)",
        "external": "external=home/'external';external.mkdir(mode=0o700);(external/'fixture').mkdir(mode=0o700)\nselected.write_text(selected.read_text()+f'[paths]\\ndata_dir=\"{external.as_posix()}\"\\n')",
        "ambiguous": "(home/'.tldw_cli-data').mkdir(mode=0o700)",
    }
    script = _DEFAULT_DIRECTORIES.replace(
        "authority.register('profile',(selected,store,data/'chat_dicts'))",
        (changes[mode] + "\n" if mode == "external" else "")
        + "authority.register('profile',(selected,store,data/'chat_dicts'))",
    ).split("from tldw_chatbook import config")[0]
    if mode != "external":
        script += changes[mode] + "\n"
    script += """
try:
 from tldw_chatbook import config
 config.get_user_data_dir()
except (bootstrap.RecoveryRequired,OSError,ValueError,SystemExit) as error:
 if isinstance(error,SystemExit):assert error.code not in (None,0)
else:raise AssertionError('unverified data directory admitted')
assert not (home/'.tldw_cli-data-root.lock').exists()
"""
    if mode == "unsafe":
        script += "assert native_os.stat(data).st_mode & 0o044\n"
    elif mode == "missing":
        script += "assert not data.exists()\n"
    elif mode == "alias":
        script += "assert data.is_symlink()\n"
    script += "print('retired and reopened')\n"
    _run(tmp_path, mode, "default-data", script=script, timeout=30)


_OPERATIONS = _SCRIPT.replace(
    "os.environ['TLDW_CONFIG_PATH']=str(selected)",
    "os.environ.pop('TLDW_CONFIG_PATH',None)\nfrom tldw_chatbook import config\n"
    "from tldw_chatbook.Backup_Recovery import storage_admission as storage\n"
    "assert not storage._raw_operations and not storage._operations\n"
    "storage._shutdown()\nassert not storage._live_leases",
).replace(
    "from tldw_chatbook import config\nassert config.get_cli_setting",
    "assert config.get_cli_setting",
).split("assert config.get_cli_setting")[0] + r'''
import sys,stat,threading
from contextlib import contextmanager
from tldw_chatbook.Backup_Recovery import config_participants,raw_participants as raw,storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import Admission
from tldw_chatbook.Utils.platform_files import fcntl
mode=sys.argv[1]
initial=selected.read_bytes();parent_stat=parent.stat()
registry_before=bootstrap._registry(root)
assert config.application_owned_config_directory(selected)==parent
if mode in ('foreign','alias','pending','cancel','uncertain'):
 if mode in ('foreign','alias'):
  backup=config._advanced_backup_path(selected)
  if mode=='alias':os.link(selected,backup)
  else:
   backup.write_bytes(b'foreign owned bytes');backup.chmod(0o600)
   authority.register('foreign-backup',(backup,))
  preserved=backup.read_bytes()
  try:config.read_cli_config_serialized()
  except (bootstrap.RecoveryRequired,OSError,ValueError):pass
  else:raise AssertionError('foreign companion was admitted')
  assert backup.read_bytes()==preserved and selected.read_bytes()==initial
 elif mode=='pending':
  from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
  replacement=home/'new-config';replacement.write_bytes(initial);replacement.chmod(0o600)
  with storage.acquire_storage(selected):
   try:authority.remap('profile',(replacement,data),0.01)
   except AdmissionTimeout:pass
   else:raise AssertionError('remap unexpectedly crossed native owner')
  try:config.read_cli_config_serialized()
  except (bootstrap.RecoveryRequired,OSError,ValueError):pass
  else:raise AssertionError('pending authority was admitted')
  assert selected.read_bytes()==initial
 elif mode=='cancel':
  ready=threading.Event();proceed=threading.Event();waiting=threading.Event();errors=[]
  acquire=storage.acquire_storage;lock=Admission._lock
  def observed_acquire(path=None,**kwargs):
   lease=acquire(path,**kwargs)
   if threading.current_thread().name=='reader' and path==selected:
    ready.set();assert proceed.wait(3)
   return lease
  @contextmanager
  def observed_lock(self,fd,name,lock_mode,*args,**kwargs):
   if threading.current_thread().name=='reader' and name=='registry.lock' and kwargs.get('cancel') is not None:waiting.set()
   with lock(self,fd,name,lock_mode,*args,**kwargs) as value:yield value
  storage.acquire_storage=observed_acquire;Admission._lock=observed_lock
  def reader():
   try:config.read_cli_config_serialized()
   except BaseException as error:errors.append(error)
  thread=threading.Thread(target=reader,name='reader');thread.start()
  try:
   assert ready.wait(3)
   with authority._directory() as fd,lock(authority,fd,'registry.lock',fcntl.LOCK_EX):
    proceed.set();assert waiting.wait(3)
    with storage._lock:
     pending=[attempt for attempt in storage._pending_acquisitions if attempt.thread is thread]
     assert pending
     for attempt in pending:attempt.cancel.set()
    thread.join(3);assert not thread.is_alive()
   assert errors and not raw._states and not storage._raw_operations
   assert not storage._live_leases and not storage._holds
  finally:storage.acquire_storage=acquire;Admission._lock=lock;proceed.set()
 elif mode=='uncertain':
  fd=None
  try:
   with config_participants.operation(config) as operation:
    fd=os.open(selected,os.O_RDONLY)
    raw._states[operation].descriptors.add(fd)
  except bootstrap.RecoveryRequired:pass
  else:raise AssertionError('unretired descriptor lost protection')
  assert operation in raw._states and storage._live_leases
  with authority._directory() as parent_fd:
   lock_fd=authority._open(parent_fd,'registry.lock',os.O_RDWR)
   try:
    try:fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:pass
    else:raise AssertionError('unretired companion released registry protection')
   finally:os.close(lock_fd)
  os.close(fd)
 print('retired and reopened');raise SystemExit(0)
if mode in ('write','snapshot','generic','registration','maintenance','unsafe','replaced','missing','unsafe_active'):
 if mode=='unsafe':parent.chmod(0o755)
 if mode in ('unsafe','replaced','missing','unsafe_active'):
  try:
   if mode in ('replaced','missing','unsafe_active'):
    with config_participants.operation(config):
     if mode=='unsafe_active':parent.chmod(0o755)
     else:
      moved=home/'old-parent';parent.rename(moved)
      if mode=='replaced':
       parent.mkdir(mode=0o700);selected.write_bytes(initial);selected.chmod(0o600)
     config._prepare_config_parent(selected)
   else:config.read_cli_config_serialized()
  except (bootstrap.RecoveryRequired,OSError,ValueError):pass
  else:raise AssertionError('unsafe parent was admitted')
  if mode in ('unsafe','unsafe_active'):assert stat.S_IMODE(parent.stat().st_mode)==0o755
  if mode=='replaced':assert not (parent/'recovery-bootstrap').exists()
  if mode=='missing':assert not parent.exists()
  print('retired and reopened');raise SystemExit(0)
 if mode=='write':
  loaded,backup=config.replace_cli_config_serialized(initial.decode()+'[appearance]\ntheme="textual-dark"\n')
  assert loaded['appearance']['theme']=='textual-dark' and backup.read_bytes()==initial
 elif mode=='snapshot':
  exported=config.export_cli_config_snapshot(timestamp='20260913_235900')
  assert exported.read_bytes()==initial
 elif mode=='generic':
  with config_participants.operation(config):
   assert config.application_owned_config_directory(selected) is None
   refusals=[]
   def unrelated():
    for outside in (parent,parent/'unrelated.txt',root,root/'admission'/'registry.json'):
     try:
      with storage.acquire_storage(outside):pass
     except bootstrap.RecoveryRequired:refusals.append(outside)
   thread=threading.Thread(target=unrelated);thread.start();thread.join(5)
   assert not thread.is_alive() and len(refusals)==4
 elif mode in ('registration','maintenance'):
  attempted=threading.Event();done=threading.Event();errors=[]
  original=Admission._lock
  @contextmanager
  def observed(self,fd,name,lock_mode,*args,**kwargs):
   if threading.current_thread().name=='contender' and (
    name=='registry.lock' and lock_mode==fcntl.LOCK_EX
    or mode=='maintenance' and name.endswith('.lease') and lock_mode==fcntl.LOCK_EX
   ):attempted.set()
   with original(self,fd,name,lock_mode,*args,**kwargs) as value:yield value
  Admission._lock=observed
  def contender():
   try:
    if mode=='registration':authority.register('foreign-lock',(selected.with_name(selected.name+'.lock'),))
    else:
     with authority.maintenance(('profile',),5):pass
   except BaseException as error:errors.append(error)
   finally:done.set()
  thread=threading.Thread(target=contender,name='contender')
  try:
   with config_participants.operation(config):
    thread.start();assert attempted.wait(3)
    assert not done.is_set(),'native contender crossed active config operation'
   assert done.wait(5);thread.join();assert not errors,errors
  finally:Admission._lock=original
else:raise AssertionError('unknown fixture mode')
assert not raw._states and not storage._raw_operations
assert config.application_owned_config_directory(selected)==parent
assert (parent.stat().st_dev,parent.stat().st_ino,stat.S_IMODE(parent.stat().st_mode))==(parent_stat.st_dev,parent_stat.st_ino,stat.S_IMODE(parent_stat.st_mode))
if mode=='write':
 import hashlib
 assert bootstrap._records(root)[1]==[dict(before[0],fingerprint=hashlib.sha256(selected.read_bytes()).hexdigest())]
else:assert bootstrap._records(root)[1]==before
if mode!='registration':assert bootstrap._registry(root)==registry_before
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "mode", ["write", "snapshot", "generic", "registration", "maintenance", "unsafe", "replaced", "missing", "unsafe_active", "foreign", "alias", "pending", "cancel", "uncertain"]
)
def test_bound_config_companion_operation_contract(tmp_path, mode):
    _run(tmp_path, mode, "native", script=_OPERATIONS, timeout=30)
