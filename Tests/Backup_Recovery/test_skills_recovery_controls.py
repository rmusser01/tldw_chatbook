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
  yield LibrarySkillsListCanvas(state=self._build_library_skills_state(),trust_posture=self._library_skills_trust_posture,id='library-skills-canvas')
 def on_mount(self,event):
  event.prevent_default()
  self._refresh_library_skills_trust_posture()
 async def on_unmount(self):pass
class Host(App):
 def on_mount(self):
  self.subject=Subject(instance)
  self.subject._library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS
  self.subject._library_skills_view='list'
  self.subject._local_source_records['skills']=(None,context)
  self.push_screen(self.subject)
async def wait_for(predicate):
 for _ in range(250):
  if predicate():return
  await asyncio.sleep(.02)
 raise AssertionError('Skills UI checkpoint did not settle')
async def run():
 host=Host()
 async with host.run_test(size=(110,44)) as pilot:
  await wait_for(lambda:hasattr(host,'subject') and bool(host.subject.query('#library-skills-trust-action')))
  subject=host.subject;button=subject.query_one('#library-skills-trust-action',Button)
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
   else:subject._library_skills_view='editor'
   await wait_for(lambda:host.screen.query_one('#skills-recovery-continue',Button).disabled)
   assert 'Selection changed' in str(host.screen.query_one('#skills-recovery-status',Static).content)
   await pilot.press('escape')
   await wait_for(lambda:host.screen is subject)
   assert not trust.trust_store.store_dir.exists()
   return
  host.screen.query_one('#skills-recovery-continue',Button).press()
  await wait_for(lambda:bool(host.screen.query('#skill-trust-bootstrap-input')))
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
 async with host.run_test(size=(110,44)):
  assert context.get('available_skills') or context.get('blocked_skills'),context
  await wait_for(lambda:hasattr(host,'subject') and bool(host.subject.query('#library-skills-trust-action')))
  button=host.subject.query_one('#library-skills-trust-action',Button)
  assert str(button.label)=='Unlock',str(button.label)
  button.press()
  await wait_for(lambda:bool(host.screen.query('#skill-trust-passphrase-input')))
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
