"""First confirmed replacement binds actual current sources before publication."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414 - native helper fixture
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_recovery_service import _LIVE_BACKUP

_BIND = r"""
import builtins,json,os,sys,tomllib
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import archive_reader,crypto,launcher
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
crypto._package_resource_root=lambda:Path(sys.argv[1])
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
from tldw_chatbook.Backup_Recovery import bootstrap
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
service=RecoveryService(default_control_root())
inspection=service.start_inspection(home/'service.tldw-backup.zip',password=None)
assert service.wait(inspection)['state']=='succeeded'
archive=service.inspection(inspection);doc=archive_reader.verify_sealed(archive)
target=service.preview_backup((selector,),options={})
assert target.complete,target.issues
actual={item.logical_id:item for item in target.items}
source_profile=doc.profile_ids[0]
current_config=next(item for item in target.items if item.owner=='config' and item.path==selector)
current_profile=current_config.logical_id.split(':')[1]
def current_key(key):
 prefix='profile:'+source_profile+':'
 return 'profile:'+current_profile+':'+key[len(prefix):] if key.startswith(prefix) else key
roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
mapping={};deferred={'agents.history','eval.definitions','tts.voices','persona.visual_identity_builtin'}
for index,(key,root) in enumerate(roots.items()):
 members=[row for row in doc.files if row.root_id==key]
 owners={row.owner_id for row in members}
 if owners & deferred:
  mapping[key]=home/('reviewed-inactive-'+str(index));continue
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
if sys.argv[2]=='drift':
 native_names=capture_service._capture_names
 def changed_config(authority,inventory):
  names=native_names(authority,inventory)
  selector.write_bytes(before+b'\n# changed after review\n')
  return names
 capture_service._capture_names=changed_config
try:
 if sys.argv[2]=='drift':
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
  assert selector.read_bytes()==before
  from tldw_chatbook.Backup_Recovery.restore_plan import recheck_targets
  recheck_targets(plan)
 assert not bootstrap._records(bootstrap.default_bootstrap_root())[0]
finally:service.close()
assert not blocked_attempts()
assert 'tldw_chatbook.app' not in sys.modules
print('actual current-source binding checked')
"""


@pytest.mark.parametrize("mode", ["same", "different", "drift"])
def test_first_binding_uses_independently_discovered_current_profile(
    tmp_path, helper_resource_root, mode
):
    _run(tmp_path, "service", "backup", script=_LIVE_BACKUP, timeout=110)
    selector = tmp_path / "config" / "config.toml"
    if mode == "different":
        current = tmp_path / "current-config" / "config.toml"
        current.parent.mkdir(mode=0o700)
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
    )
    result = subprocess.run(
        [sys.executable, "-c", _BIND, str(helper_resource_root), mode],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    (tmp_path / "first-replacement-child.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-2000:]
    assert "actual current-source binding checked" in result.stdout


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
