"""Restored-profile setup summaries read the actual durable owner requirements."""

import pytest

from Tests.Backup_Recovery.test_recovery_service import _SERVICE_ISOLATED

_SUMMARY = _SERVICE_ISOLATED.replace(
    "assert service.profiles()[0]['status']=='restoration_validated'",
    """assert service.profiles()[0]['status']=='restoration_validated'
 row=service.profiles()[0]
 assert row['requirements_checked'] and row['needs_setup'],dict(row)
 assert row['required_owners'] and row['pending_owners']==row['required_owners'],dict(row)
 from tldw_chatbook.Backup_Recovery.activation import ActivationStore
 store=ActivationStore(control/'activation')
 required=row['required_owners'];generation=row['generation']
 if sys.argv[2]=='one':
  store.approve(generation,required[0])
  row=service.profiles()[0]
  assert required[0] not in row['pending_owners']
  assert row['pending_owners']==required[1:]
  assert row['needs_setup']==bool(required[1:])
 elif sys.argv[2]=='all':
  for owner in required:store.approve(generation,owner)
  row=service.profiles()[0]
  assert row['requirements_checked'] and not row['needs_setup']
  assert row['pending_owners']==()
  assert row['status']=='restoration_validated' and not row.get('opened',False)
 elif sys.argv[2]=='damaged':
  import shutil
  shutil.rmtree(store._generation(generation))
  row=service.profiles()[0]
  assert row['status']=='recovery_required'
  assert not row['requirements_checked'] and row['needs_setup'] is None
  assert row['required_owners'] is None and row['pending_owners'] is None
 else:raise AssertionError('unknown fixture route')
 import asyncio
 from textual.app import App
 from textual.widgets import Button,Static
 from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen
 class Harness(App):
  def on_mount(self):self.push_screen(BackupRestoreScreen(service))
 async def verify_view():
  app=Harness()
  async with app.run_test(size=(90,30)) as pilot:
   await pilot.click('#backup-open-profiles')
   async with asyncio.timeout(10):
    while not app.screen.query('.backup-open-profile'):await asyncio.sleep(.03)
   text='\\n'.join(str(s.render()) for s in app.screen.query('#backup-list Static'))
   expected=('Setup requirements unavailable' if sys.argv[2]=='damaged' else
             'Needs setup' if row['needs_setup'] else 'Owner reviews complete')
   assert expected in text,text
   assert 'Opened successfully' not in text
   assert app.screen.query_one('.backup-open-profile',Button).disabled==(sys.argv[2]=='damaged')
 asyncio.run(verify_view())""",
).replace(
    "assert _launch_descriptor(profile,control).profile_id==profile\nassert archive.path",
    "assert archive.path",
)


@pytest.mark.parametrize("state", ["one", "all", "damaged"])
def test_actual_profile_requirements_are_independent_of_opened_state(tmp_path, state):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "isolated", state, script=_SUMMARY)
