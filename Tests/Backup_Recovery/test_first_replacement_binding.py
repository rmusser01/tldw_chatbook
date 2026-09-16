"""First confirmed replacement binds actual current sources before publication."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414 - component helper fixture
)
from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - installed product fixture
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_recovery_service import _LIVE_BACKUP

_BIND = r"""
import builtins,json,os,sys,tomllib
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed=Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])
assert Path(tldw_chatbook.__file__).resolve()==installed/'tldw_chatbook'/'__init__.py'
from tldw_chatbook.Backup_Recovery import archive_reader,launcher
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Backup_Recovery import recovery_service
safe_issue=recovery_service.issue_code
def diagnostic(error,**kwargs):
 import re
 code=error.args[0] if error.args and isinstance(error.args[0],str) and re.fullmatch('[a-z0-9_:]+',error.args[0]) else 'non_code'
 print('NATIVE_FAILURE',type(error).__name__,code,flush=True)
 import traceback
 print('FAILURE_BOUNDARY',[(Path(row.filename).name,row.name,row.lineno) for row in traceback.extract_tb(error.__traceback__)],flush=True)
 return safe_issue(error,**kwargs)
recovery_service.issue_code=diagnostic
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
from tldw_chatbook.Backup_Recovery import bootstrap
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
service=RecoveryService(default_control_root())
inspection=service.start_inspection(home/'service.tldw-backup.zip',password=None)
assert service.wait(inspection)['state']=='succeeded'
archive=service.inspection(inspection);doc=archive_reader.verify_sealed(archive)
mode=sys.argv[1]
if mode=='unknown':
 (selector.parent/'unowned.txt').write_text('not declared by a current owner')
if mode=='link':(selector.parent/'linked.toml').symlink_to(selector)
if mode=='hardlink':
 other=home/'hardlink-target';other.write_bytes(b'unowned hardlink')
 os.link(other,selector.parent/'linked.bin')
if mode=='private':selector.parent.chmod(0o755)
if mode=='shared':
 from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile
 root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
 other=home/'other-selector.toml';other.write_bytes(selector.read_bytes());other.chmod(0o600)
 authority.register('shared-config',(selector,other))
 bind_profile(root,other,('shared-config',),root/'admission')
target=service.preview_backup((selector,),options={})
assert target.complete,target.issues
if mode=='bootstrap':
 from tldw_chatbook.Backup_Recovery import replacement
 from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
 from tldw_chatbook.Backup_Recovery.control_records import admission_authority
 root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
 names=_capture_names(authority,target)
 try:replacement._first_config_container(selector,target,names,bootstrap._registry(root),(),(selector,),(root,))
 except ValueError as error:assert str(error)=='replacement_config_container_unverified',error
 else:raise AssertionError('recovery control parent was adopted')
 assert not bootstrap._records(root)[1]
 assert not blocked_attempts()
 print('actual current-source binding checked');service.close();raise SystemExit(0)
actual={item.logical_id:item for item in target.items}
source_profile=doc.profile_ids[0]
current_config=next(item for item in target.items if item.owner=='config' and item.path==selector)
current_profile=current_config.logical_id.split(':')[1]
def current_key(key):
 prefix='profile:'+source_profile+':'
 return 'profile:'+current_profile+':'+key[len(prefix):] if key.startswith(prefix) else key
roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
mapping={};deferred={'agents.history','eval.definitions','tts.voices','persona.visual_identity_builtin'}
inactive=home/'inactive-destinations' if mode=='abort' else home
if mode=='abort':inactive.mkdir(mode=0o700)
for index,(key,root) in enumerate(roots.items()):
 members=[row for row in doc.files if row.root_id==key]
 owners={row.owner_id for row in members}
 if owners & deferred:
  mapping[key]=inactive/('reviewed-inactive-'+str(index));continue
 if current_key(key) in actual and actual[current_key(key)].path is not None:
  mapping[key]=actual[current_key(key)].path;continue
 choices=set()
 for row in members:
  path=actual[current_key(row.logical_id)].path
  for part in Path(row.relative_path).parts:path=path.parent
  choices.add(path)
 assert len(choices)==1,(key,choices)
 mapping[key]=choices.pop()
profile=doc.profile_ids[0]
config=tomllib.loads(selector.read_text())
name=config.get('general',{}).get('users_name','default_user')
from tldw_chatbook.Backup_Recovery.profile_paths import data_base
mapping[f'profile:{profile}:paths.data_dir']=data_base(config)
print('TARGET_MAPPING',len(mapping),flush=True)
from tldw_chatbook.Backup_Recovery import replacement,capture_service
plan=service.preview_restore(inspection,mode='replace',destinations=mapping,target=target,profile_names={profile:name})
before=selector.read_bytes()
if mode=='topology':
 from contextlib import contextmanager
 from tldw_chatbook.Backup_Recovery.admission import Admission
 original_maintenance=Admission.maintenance
 @contextmanager
 def changed_topology(self,*args,**kwargs):
  with original_maintenance(self,*args,**kwargs) as session:
   (selector.parent/'unowned-late.txt').write_text('external arrival after proof')
   yield session
 Admission.maintenance=changed_topology
if sys.argv[1]=='drift':
 native_names=capture_service._capture_names
 def changed_config(authority,inventory):
  names=native_names(authority,inventory)
  selector.write_bytes(before+b'\n# changed after review\n')
  return names
 capture_service._capture_names=changed_config
try:
 if mode in {'unknown','private','shared','protected','topology','bootstrap','link','hardlink'}:
  protected=(selector.parent,) if mode=='protected' else ()
  try:replacement._ensure_first_bindings(plan,(selector,),bootstrap.default_bootstrap_root(),Event(),protected=protected)
  except ValueError as error:
   expected='target_changed' if mode=='topology' else 'replacement_config_container_unverified'
   assert str(error)==expected,error
  else:raise AssertionError('unverified config container was enrolled')
  assert not any(row['selector']==str(selector) for row in bootstrap._records(bootstrap.default_bootstrap_root())[1])
  assert selector.read_bytes()==before
  if mode=='topology':
   Admission.maintenance=original_maintenance
   current=service.preview_backup((selector,),options={})
   fresh=service.preview_restore(inspection,mode='replace',destinations=mapping,target=current,profile_names={profile:name})
   try:replacement._ensure_first_bindings(fresh,(selector,),bootstrap.default_bootstrap_root(),Event())
   except ValueError as error:assert str(error)=='replacement_config_container_unverified',error
   else:raise AssertionError('a failed registration bypassed fresh closure proof')
   assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
   late=selector.parent/'unowned-late.txt'
   assert late.read_text()=='external arrival after proof';late.unlink()
   current=service.preview_backup((selector,),options={})
   fresh=service.preview_restore(inspection,mode='replace',destinations=mapping,target=current,profile_names={profile:name})
   replacement._ensure_first_bindings(fresh,(selector,),bootstrap.default_bootstrap_root(),Event())
   bound=bootstrap._records(bootstrap.default_bootstrap_root())[1]
   assert len(bound)==1 and str(selector.parent) in bound[0]['roots']
 elif sys.argv[1]=='drift':
  try:replacement._ensure_first_bindings(plan,(selector,),bootstrap.default_bootstrap_root(),Event())
  except ValueError as error:assert str(error)=='target_changed',error
  else:raise AssertionError('changed config was accepted')
  assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
 else:
  replacement._ensure_first_bindings(plan,(selector,),bootstrap.default_bootstrap_root(),Event())
  bindings=bootstrap._records(bootstrap.default_bootstrap_root())[1]
  selected=next(row for row in bindings if row['selector']==str(selector))
  assert selected['namespaces'] and 'bootstrap.unbound' not in selected['namespaces']
  assert bootstrap._binding(selector,bindings,bootstrap._registry(bootstrap.default_bootstrap_root())) is not None
  if mode in {'companions','abort'}:
   assert str(selector.parent) in selected['roots'], 'current config companions are not enrolled'
  assert selector.read_bytes()==before
  from tldw_chatbook.Backup_Recovery.restore_plan import recheck_targets
  recheck_targets(plan)
  if mode=='abort':
   import hashlib
   from tldw_chatbook.Backup_Recovery.journal import Journal
   from tldw_chatbook.Backup_Recovery.staging import stage_restore
   source_bytes={str(item.path):hashlib.sha256(item.path.read_bytes()).hexdigest() for item in target.items if item.path is not None and item.status=='included' and item.path.is_file()}
   candidate=stage_restore(archive,plan,home/'abort-stage',Event())
   cancellation=Event();native_prepare=Journal.prepare_publication
   def cancel_prepared(self,*args,**kwargs):
    result=native_prepare(self,*args,**kwargs);cancellation.set();return result
   Journal.prepare_publication=cancel_prepared
   try:
    try:replacement.replace(plan,candidate,control_root=home/'abort-control',rollback_password=b'private-test-password',cancel=cancellation)
    except InterruptedError:pass
    else:raise AssertionError('prepared cancellation did not stop replacement')
   finally:Journal.prepare_publication=native_prepare
   pending,prior=bootstrap._records(bootstrap.default_bootstrap_root())
   assert len(pending)==1 and prior==bindings
   operation=pending[0]['operation_id'];journal=Journal(home/'abort-control',operation)
   with journal._locked(exclusive=False) as fd:events=[row.event for row in journal._records(fd)]
   assert events==['candidate_staged','prepared'],events
   assert replacement.recover_replacement(operation,control_root=home/'abort-control',action='abort',rollback_password=None,cancel=Event())=='aborted'
   assert bootstrap._records(bootstrap.default_bootstrap_root())[1]==bindings
   assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in source_bytes.items())
   with journal._locked(exclusive=False) as fd:assert journal._records(fd)[-1].event=='prepublication_aborted'
 assert not bootstrap._records(bootstrap.default_bootstrap_root())[0]
finally:service.close()
assert not blocked_attempts()
assert 'tldw_chatbook.app' not in sys.modules
print('actual current-source binding checked')
"""


@pytest.mark.parametrize(
    "mode",
    [
        "same",
        "different",
        "drift",
        "companions",
        "abort",
        "unknown",
        "private",
        "shared",
        "protected",
        "topology",
        "bootstrap",
        "link",
        "hardlink",
    ],
)
def test_first_binding_uses_independently_discovered_current_profile(
    tmp_path, native_package, mode
):
    script = _LIVE_BACKUP
    if mode in {"companions", "abort"}:
        script = (
            r"""
import json,os
from pathlib import Path
selected=Path(os.environ['TLDW_CONFIG_PATH'])
data=selected.parent/'data';data.mkdir(mode=0o700)
selected.write_text('[general]\nusers_name="default_user"\n[paths]\ndata_dir='+json.dumps(str(data))+'\n')
selected.chmod(0o600)
"""
            + script
        )
    _run(
        tmp_path,
        "service",
        "backup",
        script=script,
        timeout=110,
        installed_package=native_package,
    )
    selector = tmp_path / "config" / "config.toml"
    if mode not in {"same", "drift", "companions", "abort"}:
        current = (
            tmp_path / "home" / ".config" / "tldw_cli" / "config.toml"
            if mode == "bootstrap"
            else tmp_path / "current-config" / "config.toml"
        )
        current.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        current.write_bytes(selector.read_bytes())
        current.chmod(0o600)
        selector = current
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join(
            (str(native_package), str(Path(__file__).resolve().parents[2]))
        ),
        TLDW_TEST_INSTALLED_PACKAGE=str(native_package),
    )
    result = subprocess.run(
        [sys.executable, "-c", _BIND, mode],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    (tmp_path / "first-replacement-child.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-2000:]
    assert "actual current-source binding checked" in result.stdout
    if mode in {"companions", "abort"}:
        # Fixed private CLI child; no shell or caller-supplied program.
        reopened = subprocess.run(  # nosec B603
            [sys.executable, "-c", _READ_BOUND],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        (tmp_path / "bound-config-cli.log").write_text(
            reopened.stdout + reopened.stderr
        )
        assert reopened.returncode == 0, (
            reopened.stdout[-2000:] + reopened.stderr[-3000:]
        )
        assert "fresh CLI retained current config scope" in reopened.stdout


_READ_BOUND = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed=Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])
assert Path(tldw_chatbook.__file__).resolve()==installed/'tldw_chatbook'/'__init__.py'
from tldw_chatbook.Backup_Recovery import bootstrap
selector=Path(os.environ['TLDW_CONFIG_PATH'])
before=bootstrap._records(bootstrap.default_bootstrap_root())[1]
from tldw_chatbook.cli import main_cli_runner
sys.argv=['tldw-chatbook','--help']
try:main_cli_runner()
except SystemExit as error:assert error.code in (None,0)
from tldw_chatbook import config
assert config._get_effective_config_path()==selector
assert bootstrap._records(bootstrap.default_bootstrap_root())[1]==before
assert not blocked_attempts()
print('fresh CLI retained current config scope')
"""


@pytest.mark.parametrize("scope", ["held", "missing_unbound", "missing_source", "fake"])
def test_first_binding_requires_actual_held_unbound_and_source_scope(tmp_path, scope):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        UNBOUND_NAMESPACE,
        admission_authority,
        bind_profile,
    )

    root = tmp_path / "bootstrap"
    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    selector = live / "config.toml"
    selector.write_text('[general]\nusers_name="local"\n')
    selector.chmod(0o600)
    authority = admission_authority(root)
    authority.register("current", (live,))
    if scope == "fake":
        with pytest.raises(ValueError, match="enrollment_session_required"):
            bind_profile(
                root, selector, ("current",), root / "admission", session=object()
            )
    else:
        names = {
            "held": (UNBOUND_NAMESPACE, "current"),
            "missing_unbound": ("current",),
            "missing_source": (UNBOUND_NAMESPACE,),
        }[scope]
        with authority.maintenance(names, 2) as session:
            if scope == "held":
                bind_profile(
                    root, selector, ("current",), root / "admission", session=session
                )
            else:
                with pytest.raises(ValueError, match="enrollment_session_scope"):
                    bind_profile(
                        root,
                        selector,
                        ("current",),
                        root / "admission",
                        session=session,
                    )
    assert bool(bootstrap._records(root)[1]) == (scope == "held")


def test_existing_conflicting_binding_is_not_replaced(
    tmp_path, monkeypatch, helper_resource_root
):
    from dataclasses import replace
    from threading import Event

    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import bootstrap, crypto, replacement
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        root = bootstrap.default_bootstrap_root()
        original_binding = bootstrap._records(root)[1]
        original_payloads = (source.read_bytes(), selector.read_bytes())
        alternate = tmp_path / "other-existing-source"
        alternate.mkdir(mode=0o700)
        admission_authority(root).remap("profile", (alternate,), 2)
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        with pytest.raises(ValueError, match="replacement_binding_changed"):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"test-only-password",
                cancel=Event(),
            )
        assert bootstrap._records(root)[1] == original_binding
        assert not bootstrap._records(root)[0]
        assert (source.read_bytes(), selector.read_bytes()) == original_payloads
