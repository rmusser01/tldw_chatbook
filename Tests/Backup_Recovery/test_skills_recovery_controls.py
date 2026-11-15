"""Actual mounted Library Skills recovery review controls."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_skills_recovery_review import _SETUP


def test_recovered_posture_is_passive_and_distinct_from_unlock(tmp_path):
    script = (
        _SETUP
        + r"""
assert trust.recovery_posture()=='recovery_review'
assert not trust.trust_store.store_dir.exists()
assert not activation.allowed(witness['generation'],'skills')
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "skills", "posture", script=script)


_UI = (
    _SETUP
    + r"""
from types import SimpleNamespace
from textual.app import App
from textual.widgets import Button,Input,Static
from tldw_chatbook.css.Themes.themes import ThemeVariableDefaultsMixin
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_skills_canvas import LibrarySkillsListCanvas
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
action=sys.argv[2]
if action=='foreign':
 trust.trust_store.store_dir.mkdir(mode=0o700)
 (trust.trust_store.store_dir/'foreign').write_bytes(b'preserve unbound evidence')
if action=='resume':
 real_approve=ActivationStore.approve
 def interrupt_skills(self,generation,owner):
  if owner=='skills':raise OSError('interrupt before owner approval')
  return real_approve(self,generation,owner)
 ActivationStore.approve=interrupt_skills
 try:trust.trust_reviewed_recovery(trust.capture_recovery_review(),'fresh-ui-passphrase')
 except OSError:pass
 else:raise AssertionError('setup interruption did not occur')
 finally:ActivationStore.approve=real_approve
 assert trust.trust_store.has_manifest() and not activation.allowed(witness['generation'],'skills')
if action=='empty':
 import shutil
 shutil.rmtree(restored/'skills')
scope=SkillsScopeService(local_service=local)
context=asyncio.run(scope.get_context(mode='local'))
notices=[]
instance=SimpleNamespace(app_config={},local_skill_trust_service=trust,skills_scope_service=scope,notify=lambda message,**kwargs:notices.append(message))
class Subject(LibraryScreen):
 def compose(self):
  yield LibrarySkillsListCanvas(state=self._build_library_skills_state(),trust_posture=self._skills_state.trust_posture,id='library-skills-canvas')
 def on_mount(self,event):
  event.prevent_default()
  self._refresh_library_skills_trust_posture()
 async def on_unmount(self):pass
# TASK-33641.7: the Library sheet names $tldw-boundary, which no tcss defines;
# the mixin supplies the guard names, as it does for TldwCli.
class Host(ThemeVariableDefaultsMixin,App):
 def on_mount(self):
  self.subject=Subject(instance)
  self.subject._library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS
  self.subject._skills_state.view='list'
  self.subject._local_source_records['skills']=(None,context)
  self.push_screen(self.subject)
# A wall-clock deadline, not a poll count: under a parallel run 250 polls ran
# out waiting for the review modal. 20 s stays inside the 45 s process limit.
async def wait_for(predicate):
 clock=asyncio.get_running_loop().time;deadline=clock()+20
 while not predicate():
  assert clock()<deadline,'Skills UI checkpoint did not settle'
  await asyncio.sleep(.02)
# Each posture load (on mount, then after the source snapshot) recomposes the
# header, and a press on a Button a later load replaced is dropped with it: in
# 3 of 400 [empty] runs the review never opened (PR #3169's CI run 2). So wait
# for no worker left and let the last recompose land before pressing.
async def settled_action(pilot,host):
 await wait_for(lambda:hasattr(host,'subject') and bool(host.subject.query('#library-skills-trust-action')) and not host.workers)
 await pilot.pause()
 return host.subject.query_one('#library-skills-trust-action',Button)
async def run():
 host=Host()
 async with host.run_test(size=(110,44)) as pilot:
  button=await settled_action(pilot,host);subject=host.subject
  assert str(button.label)=='Review restored skills',str(button.label)
  button.press()
  await wait_for(lambda:bool(host.screen.query('#skills-recovery-files')))
  assert ('No current Skills bundles.' if action=='empty' else 'demo') in str(host.screen.query_one('#skills-recovery-files',Static).content)
  assert 'demo' in str(host.screen.query_one('#skills-recovery-history',Static).content)
  assert trust.trust_store.store_dir.exists()==(action in ('resume','foreign'))
  assert not subject.query('#library-skills-trust-reset')
  if action=='cancel-review':
   await pilot.press('escape')
   await wait_for(lambda:host.screen is subject)
   assert not trust.trust_store.store_dir.exists()
   return
  if action in ('service','root','navigation'):
   if action=='service':instance.local_skill_trust_service=service(restored)
   elif action=='root':trust.skills_dir=restored/'other-skills'
   else:subject._skills_state.view='editor'
   await wait_for(lambda:host.screen.query_one('#skills-recovery-continue',Button).disabled)
   assert 'Selection changed' in str(host.screen.query_one('#skills-recovery-status',Static).content)
   await pilot.press('escape')
   await wait_for(lambda:host.screen is subject)
   assert not trust.trust_store.store_dir.exists()
   return
  host.screen.query_one('#skills-recovery-continue',Button).press()
  # The Submit button mounts one container level after the inputs.
  await wait_for(lambda:bool(host.screen.query('#skill-trust-bootstrap-submit')))
  if action=='cancel-passphrase':
   await pilot.press('escape')
   await wait_for(lambda:host.screen is subject)
   assert not trust.trust_store.store_dir.exists()
   return
  if action=='changed':(restored/'skills'/'demo'/'SKILL.md').write_text('changed after displayed review')
  if action=='service-passphrase':instance.local_skill_trust_service=service(restored)
  host.screen.query_one('#skill-trust-bootstrap-input',Input).value='fresh-ui-passphrase'
  host.screen.query_one('#skill-trust-bootstrap-confirm-input',Input).value='mismatched-passphrase'
  host.screen.query_one('#skill-trust-bootstrap-submit',Button).press()
  await wait_for(lambda:bool(str(host.screen.query_one('#skill-trust-bootstrap-error',Static).content)))
  assert not activation.allowed(witness['generation'],'skills')
  host.screen.query_one('#skill-trust-bootstrap-confirm-input',Input).value='fresh-ui-passphrase'
  host.screen.query_one('#skill-trust-bootstrap-submit',Button).press()
  if action=='foreign':
   await wait_for(lambda:bool(notices))
   assert (trust.trust_store.store_dir/'foreign').read_bytes()==b'preserve unbound evidence'
   return
  if action=='changed':
   await wait_for(lambda:bool(notices))
   assert 'Skills changed after review' in notices[-1],notices
   assert not trust.trust_store.store_dir.exists()
   return
  if action=='service-passphrase':
   await wait_for(lambda:host.screen is subject)
   await asyncio.sleep(.1)
   assert not trust.trust_store.store_dir.exists()
   return
  await wait_for(lambda:activation.allowed(witness['generation'],'skills'))
  if action!='empty':trust.ensure_skill_trusted('demo')
  assert not trust.script_execution_granted('demo')
  assert not activation.allowed(witness['generation'],'config')
  assert not activation.allowed(witness['generation'],'mcp.local')
  assert all((historical/name).read_bytes()==data for name,data in history.items())
asyncio.run(run())
assert not activation.allowed(witness['generation'],'config')
if action not in ('approve','empty','resume'):
 assert not activation.allowed(witness['generation'],'skills')
 assert trust.trust_store.store_dir.exists()==(action=='foreign')
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert before=={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_mounted_library_requires_review_then_fresh_passphrase(tmp_path):
    _run(tmp_path, "skills", "approve", script=_UI, timeout=45)


@pytest.mark.parametrize(
    "action",
    [
        "cancel-review",
        "cancel-passphrase",
        "changed",
        "empty",
        "resume",
        "foreign",
        "service",
        "root",
        "navigation",
        "service-passphrase",
    ],
)
def test_actual_mounted_skills_review_preserves_cancellation_and_current_scope(
    tmp_path, action
):
    _run(tmp_path, "skills", action, script=_UI, timeout=45)


def test_actual_mounted_ordinary_library_keeps_existing_unlock(tmp_path):
    script = (
        _SETUP[: _SETUP.index("before={")]
        + r"""
trust=service(store_root)
local=LocalSkillsService(store_dir=store_root,trust_service=trust)
assert trust.recovery_posture()=='locked'
history={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
"""
        + _UI[
            _UI.index("from types import SimpleNamespace") : _UI.index(
                "async def run():"
            )
        ]
        + r"""
async def run():
 host=Host()
 async with host.run_test(size=(110,44)) as pilot:
  assert context.get('available_skills') or context.get('blocked_skills'),context
  button=await settled_action(pilot,host)
  assert str(button.label)=='Unlock',str(button.label)
  button.press()
  # As above: Submit mounts one container level after the input.
  await wait_for(lambda:bool(host.screen.query('#skill-trust-passphrase-submit')))
  host.screen.query_one('#skill-trust-passphrase-input',Input).value='old-passphrase'
  host.screen.query_one('#skill-trust-passphrase-submit',Button).press()
  await wait_for(lambda:trust.recovery_posture()=='ready')
  assert trust.script_execution_granted('demo')
  assert history=={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
asyncio.run(run())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "skills", "ordinary", script=script, timeout=45)


_TRUST_ARCHIVE = r"""
import asyncio,json,os,tomllib
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install();os.umask(0o077)
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
from tldw_chatbook.Backup_Recovery import bootstrap,credentials
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
store=KeyringServerCredentialStore(keyring_backend=FakeKeyring())
credentials._credential_store=lambda:store
source=Path(os.environ['TLDW_CONFIG_PATH'])
data=Path(os.environ['XDG_DATA_HOME'])/'profile';data.mkdir(mode=0o700)
source.write_text('[general]\nusers_name="Local"\n[paths]\ndata_dir='+json.dumps(str(data))+'\n');source.chmod(0o600)
# The installed required stores, as a real profile has them (TASK-40135's PREPARE).
import subprocess,sys
prepare=('from pathlib import Path;import os;from Tests import network_guard;network_guard.install();'
         'from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies;'
         'from tldw_chatbook.Backup_Recovery.capture_service import preview_capture;'
         "_populate_required_dependencies(preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True}));"
         'assert not network_guard.blocked_attempts()')
done=subprocess.run([sys.executable,'-c',prepare],capture_output=True,text=True,timeout=120)
assert done.returncode==0,done.stderr[-4000:]
root=user_data_dir(tomllib.loads(source.read_text()))/'skills';trust_root=root/'trust'
trust=SkillTrustService(skills_dir=root/'skills',trust_store=SkillTrustStore(trust_root,FileSkillTrustGenerationMarkerStore(trust_root/'generation_marker.json',store_dir=trust_root)))
trust.unlock_with_passphrase('old-passphrase',salt=b'6'*32)
local=LocalSkillsService(store_dir=root,trust_service=trust)
asyncio.run(local.create_skill(name='demo',content='---\nname: demo\ndescription: test\n---\n# Demo\n'))
trust.bootstrap_trust()
assert any(trust_root.rglob('*.json')),list(trust_root.rglob('*'))
admission_authority(bootstrap.default_bootstrap_root())
service=RecoveryService(Path.home()/'recovery-control')
GROUP=('tools',);password=b'skill-trust-test';destination=Path.home()/'skills.tldw-backup.zip.age'
def finished(operation):
 result=service.wait(operation)
 assert result['state']=='succeeded',dict(result)
 return result
try:
 acknowledged=()
 for _ in range(3):
  options={'data_groups':GROUP,'encrypted':True,'credential_mode':'include','acknowledged_credential_issues':acknowledged}
  preview=service.preview_backup((source,),options=options)
  result=service.wait(service.start_backup((source,),preview.scope_digest,destination,options=options,password=password))
  if result['state']=='succeeded':break
  assert result['issues']==('review_required',),dict(result)
  acknowledged=tuple(sorted(set(acknowledged)|set(result['review_issues'])))
 else:raise AssertionError('backup review did not settle')
 trust_issues=[i for i in acknowledged if i.startswith('credential_skill_trust_manual_unlock_required:')]
 assert trust_issues,acknowledged  # still disclosed, and acknowledged at capture
 inspection=service.start_inspection(destination,password=password);finished(inspection)
 summary=service.summary(inspection)
 profile=summary['profile_ids'][0]
 try:
  plan=service.preview_restore(inspection,mode='replace',profile_bases={},target_configs={profile:source},external_destinations={},profile_names={},target=service.preview_backup((source,),options={}),data_groups=GROUP)
 except ValueError as error:
  refused=str(error)
 else:
  refused=None
 print('consistency',summary['consistency'],'replace refused:',refused)
 assert (summary['consistency'],refused)==('coherent',None),(summary['consistency'],refused)
 assert plan.mode=='replace'
 assert set(trust_issues)<=set(summary['report']),summary['report']
finally:
 service.close()
assert not blocked_attempts()
print('retired and reopened')
"""


def test_include_mode_archive_with_skill_trust_files_is_not_partial(tmp_path):
    """TASK-41123: an include-mode backup holding the skills trust store
    discloses that the trust keys' keyring cache is left out
    (`credential_skill_trust_manual_unlock_required`). That disclosure is
    manual, like TASK-40135's Personal Context link keys: it must not make the
    archive `partial`, which `plan_restore` refuses for a replace restore
    (`partial_replacement`). It is still acknowledged at capture and listed in
    the archive's report."""
    _run(tmp_path, "skills", "trust-archive", script=_TRUST_ARCHIVE, timeout=150)
