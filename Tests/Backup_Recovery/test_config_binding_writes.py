"""Only the installed config writer may advance an unchanged recovery binding."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_recovery_service import _SERVICE_ISOLATED

_CHILD = r"""
import hashlib,json,os,sys
from pathlib import Path
from tldw_chatbook.cli import main_cli_runner
profile,control,mode=sys.argv[1:]
control=Path(control)
sys.argv=['tldw-chatbook','--recovery-profile',profile,'--recovery-control-root',str(control),'--help']
try:main_cli_runner()
except SystemExit as error:assert error.code in (0,None)
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage
from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor,profile_requirements
selected=config.get_cli_config_path();root=bootstrap.default_bootstrap_root()
name='profile-'+bootstrap._key(str(selected))+'.json'
record_path=root/name
association=root/('activation-'+bootstrap._key(str(selected))+'.json')
before=json.loads(record_path.read_bytes());activation_before=association.read_bytes()
requirements_before=profile_requirements(profile,control)
config_before=selected.read_bytes()
original=config.atomic_private_write_text
def injected(*args,**kwargs):
 if mode=='failed_write':raise OSError('synthetic write failure')
 result=original(*args,**kwargs)
 if mode=='interrupted':raise InterruptedError('synthetic interruption after publication')
 if mode in ('published_inode_swap','authority_inode_swap'):
  target=selected if mode=='published_inode_swap' else record_path
  other=target.with_name('synthetic-replacement')
  other.write_bytes(target.read_bytes());other.chmod(0o600)
  os.replace(other,target)
 if mode=='generation_race':
  changed=dict(before);changed['activation']=dict(before['activation'],generation='foreign-generation')
  record_path.write_text(json.dumps(changed))
 return result
if mode in ('failed_write','interrupted','published_inode_swap','authority_inode_swap','generation_race'):
 config.atomic_private_write_text=injected
if mode=='external_preimage':
 selected.write_bytes(config_before+b'\n# external edit\n')
if mode=='provider':
 result=config.save_setting_to_cli_config('api_settings.openai','api_key','synthetic-reconnect-value')
elif mode=='outside_path':
 outside=Path.home()/'outside.db';outside.write_bytes(b'outside')
 result=config.save_setting_to_cli_config('database','media_db_path',str(outside))
else:
 result=config.save_setting_to_cli_config('general','default_theme','textual-light')
after=json.loads(record_path.read_bytes())
if mode in ('theme','provider','outside_path'):
 assert result,'installed settings write did not succeed'
 assert after==dict(before,fingerprint=hashlib.sha256(selected.read_bytes()).hexdigest()),'binding did not follow successful owned write'
 assert after['fingerprint']!=before['fingerprint']
 assert association.read_bytes()==activation_before
 assert profile_requirements(profile,control)==requirements_before
 assert _launch_descriptor(profile,control).profile_id==profile
 if mode=='outside_path':
  try:storage.acquire_storage(outside)
  except bootstrap.RecoveryRequired as error:assert str(error)=='storage_scope_not_enrolled',error
  else:raise AssertionError('config write granted outside storage')
else:
 assert after['fingerprint']==before['fingerprint'],'unverified mutation refreshed binding'
 if mode!='external_preimage':assert not result,'failed/racing write claimed full success'
 if mode=='failed_write':assert selected.read_bytes()==config_before
 else:
  try:_launch_descriptor(profile,control)
  except (ValueError,OSError):pass
  else:raise AssertionError('unverified changed config remained launchable')
print('verified config binding outcome',mode)
"""


@pytest.mark.parametrize(
    "mode",
    (
        "theme",
        "provider",
        "outside_path",
        "external_preimage",
        "published_inode_swap",
        "authority_inode_swap",
        "generation_race",
        "failed_write",
        "interrupted",
    ),
)
def test_owned_config_write_preserves_only_verified_binding(tmp_path, mode):
    launch = r"""
 import subprocess
 child=subprocess.run([sys.executable,'-c',CHILD,profile,str(control),sys.argv[2]],capture_output=True,text=True,timeout=30)
 assert child.returncode==0,child.stderr[-6000:]+child.stdout[-1000:]
 print(child.stdout)
"""
    script = (
        "CHILD="
        + repr(_CHILD)
        + "\n"
        + _SERVICE_ISOLATED.replace(
            " assert service.profiles()[0]['status']=='restoration_validated'",
            " assert service.profiles()[0]['status']=='restoration_validated'" + launch,
        ).replace(
            "assert _launch_descriptor(profile,control).profile_id==profile\nassert archive.path",
            "assert archive.path",
        )
    )
    _run(tmp_path, "isolated", mode, script=script, timeout=50)


@pytest.mark.parametrize("explicit_selector", [False, True])
def test_default_binding_survives_owned_config_write_before_first_activation(
    tmp_path, explicit_selector
):
    from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT

    script = _SCRIPT.replace(
        "print('retired and reopened')",
        "registry_before=bootstrap._registry(root)\n"
        "assert before[0].get('activation') is None\n"
        "assert config.save_setting_to_cli_config('general','default_theme','textual-light')\n"
        "after=bootstrap._records(root)[1]\n"
        "assert bootstrap._binding(selected,after,bootstrap._registry(root)) is not None,"
        "'ordinary installed write invalidated the default binding'\n"
        "assert after[0]['roots']==before[0]['roots']\n"
        "assert after[0]['namespaces']==before[0]['namespaces']\n"
        "assert after[0].get('activation') is None\n"
        "assert bootstrap._registry(root)==registry_before\n"
        "print('retired and reopened')",
    )
    if not explicit_selector:
        script = script.replace(
            "os.environ['TLDW_CONFIG_PATH']=str(selected)",
            "os.environ.pop('TLDW_CONFIG_PATH',None)",
        )
    _run(tmp_path, "default-binding", "owned-write", script=script, timeout=30)


def test_disjoint_profile_write_does_not_invalidate_owned_publication(tmp_path):
    from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT

    setup = r"""
other_parent=home/'other-profile';other_parent.mkdir(mode=0o700)
other=other_parent/'config.toml';other_data=home/'other-data'
other_data.mkdir(mode=0o700);(other_data/'other').mkdir(mode=0o700)
other.write_text(f'[general]\nusers_name="other"\n[paths]\ndata_dir="{other_data.as_posix()}"\n')
other.chmod(0o600);other.with_name(other.name+'.lock').touch(mode=0o600)
authority.register('other-profile',(other,other_data))
bind_profile(root,other,('other-profile',),root/'admission')
"""
    concurrent_write = r"""
import subprocess,sys
registry_before=bootstrap._registry(root)
original=config.atomic_private_write_text
child_code='from tldw_chatbook import config; assert config.save_setting_to_cli_config("general","default_theme","textual-light")'
def interleaved(*args,**kwargs):
 child_env=os.environ.copy();child_env['TLDW_CONFIG_PATH']=str(other)
 child=subprocess.run([sys.executable,'-c',child_code],env=child_env,capture_output=True,text=True,timeout=20)
 assert child.returncode==0,child.stderr[-2500:]
 return original(*args,**kwargs)
config.atomic_private_write_text=interleaved
assert config.save_setting_to_cli_config('general','default_theme','textual-dark')
rows=bootstrap._records(root)[1]
for selector in (selected,other):
 assert bootstrap._binding(selector,rows,bootstrap._registry(root)) is not None
assert bootstrap._registry(root)==registry_before
assert [(r['selector'],r['roots'],r['namespaces']) for r in rows]==[(r['selector'],r['roots'],r['namespaces']) for r in before]
print('retired and reopened')
"""
    script = _SCRIPT.replace(
        "before=bootstrap._records(root)[1]",
        setup + "\nbefore=bootstrap._records(root)[1]",
    ).replace("print('retired and reopened')", concurrent_write)
    _run(tmp_path, "disjoint-profile", "owned-write", script=script, timeout=40)
