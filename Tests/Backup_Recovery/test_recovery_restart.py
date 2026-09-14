"""Replacement uses a fresh recovery UI after the normal app shuts down."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - native installed fixture
)


@pytest.mark.parametrize(
    "source,target",
    [("relative.zip", "/tmp/config.toml"), ("/tmp/archive.zip", "relative.toml")],
)
def test_restart_hints_require_absolute_local_paths(source, target):
    from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart

    with pytest.raises(ValueError, match="invalid_recovery_restart"):
        RecoveryRestart(Path(source), Path(target))


def test_restart_exec_uses_fixed_fresh_interpreter_and_filtered_environment(
    monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import recovery_restart
    from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart, restart

    request = RecoveryRestart(tmp_path / "archive.zip", tmp_path / "config.toml")
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "private-fixture-key")
    calls = []
    monkeypatch.setattr(
        recovery_restart,
        "os",
        SimpleNamespace(
            name="posix",
            execve=lambda executable, argv, env: calls.append((executable, argv, env)),
        ),
    )
    restart(request)
    executable, argv, environment = calls[0]
    assert executable == sys.executable
    assert argv[1:3] == ["-P", "-c"]
    assert argv[-3:] == [str(request.archive), str(request.target_config), "inspect"]
    assert "TEST_PROVIDER_API_KEY" not in environment
    assert environment["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert "private-fixture-key" not in repr(calls)


def test_actual_restart_keeps_launching_package_ahead_of_shadow_cwd(tmp_path):
    """An inherited working directory cannot substitute the recovery package."""
    import os
    import subprocess  # nosec B404 - fixed interpreter and harmless test probe.
    import sys

    import tldw_chatbook

    trusted_package = Path(tldw_chatbook.__file__).resolve()
    shadow = tmp_path / "tldw_chatbook"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("# Harmless competing package.\n")
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment.update(
        HOME=str(home), USERPROFILE=str(home),
        PYTHONPATH=str(trusted_package.parent.parent),
    )
    # The first interpreter loads the trusted restart implementation. Only its
    # UI entry is replaced; restart's real exec/CreateProcess and env stay live.
    script = """
import sys
from pathlib import Path
import tldw_chatbook
from tldw_chatbook.Backup_Recovery import recovery_restart
assert Path(tldw_chatbook.__file__).resolve() == Path(sys.argv[1])
recovery_restart._ENTRY = (
    'from pathlib import Path; import tldw_chatbook; '
    'print(Path(tldw_chatbook.__file__).resolve(), flush=True)'
)
recovery_restart.restart(recovery_restart.RecoveryRestart(None, Path(sys.argv[2])))
"""
    result = subprocess.run(  # nosec B603 - fixed code and fixture paths, no shell.
        [sys.executable, "-P", "-c", script, str(trusted_package), str(home / "config.toml")],
        cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == trusted_package


_MINIMAL = r"""
import asyncio,sys,os
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart
from tldw_chatbook.Backup_Recovery.launcher import recovery_app
from textual.widgets import Input,Select,Button
selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_bytes(b'invalid existing configuration')
selector.chmod(0o600)
source=Path.home()/'chosen.zip';source.write_bytes(b'uninspected archive')
async def main():
 app=recovery_app('replacement_requested',restart_request=RecoveryRestart(source,selector))
 async with app.run_test(size=(100,36)) as pilot:
  screen=app.screen
  assert screen.query_one('#backup-source',Input).value==str(source)
  assert screen.query_one('#backup-target-config',Input).value==str(selector)
  assert screen.query_one('#backup-restore-mode',Select).value=='replace'
  assert screen._inspection_id is None
  assert screen.query_one('#backup-start-restore',Button).disabled
  assert not screen.query_one('#backup-restart').display
  assert app.recovery_service.current() is None
 assert selector.read_bytes()==b'invalid existing configuration'
 assert source.read_bytes()==b'uninspected archive'
 assert 'tldw_chatbook.app' not in sys.modules
 assert 'tldw_chatbook.config' not in sys.modules
 assert not blocked_attempts(),blocked_attempts()
asyncio.run(main())
print('retired and reopened')
"""


def test_fresh_recovery_ui_requires_new_inspection_and_preserves_invalid_config(
    tmp_path,
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "restart", "minimal", script=_MINIMAL)


_HANDOFF_BODY = r"""
from textual.widgets import Input,Button,Select
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart
from threading import Event
source=sealed(Path.home()).path
async def main():
 app=TldwCli()
 app._recovery_restart_available=sys.argv[2]!='unsupported'
 app.app_config['_first_run']=False
 app.app_config.setdefault('first_run',{})['setup_completed']=True
 release=Event()
 async with app.run_test(size=(100,36)) as pilot:
  await pilot.press('f9');await pilot.pause()
  app.screen.query_one('#settings-backup-restore',Button).focus()
  await pilot.press('enter');await pilot.pause()
  screen=app.screen
  if sys.argv[2]=='later':
   screen._show_mode('copies')
   screen._select_rollback(Button.Pressed(Button('old copy',name='old-copy')))
   screen.query_one('#backup-later-target',Input).value=str(selector)
   password_field=screen.query_one('#backup-copy-password',Input)
   password_field.value='not-carried-test-password'
   await pilot.pause()
   assert screen.query_one('#backup-later-review',Button).disabled
   screen.query_one('#backup-later-restart',Button).focus();await pilot.press('enter')
  elif sys.argv[2]=='handoff':
   await pilot.click('#backup-open-inspect')
   screen.query_one('#backup-source',Input).value=str(source)
   await pilot.click('#backup-inspect')
   async with asyncio.timeout(15):
    while screen._inspection_id is None:await asyncio.sleep(.02)
   screen.query_one('#backup-restore-mode',Select).value='replace'
   screen.query_one('#backup-target-config',Input).value=str(selector)
   password_field=screen.query_one('#backup-rollback-password',Input)
   password_field.value='not-carried-test-password'
   await pilot.pause()
   assert screen.query_one('#backup-review-restore',Button).disabled
   screen.query_one('#backup-restart',Button).focus();await pilot.press('enter')
  else:
   if sys.argv[2]=='busy':
    app.recovery_service._start('fixture_read',lambda operation,cancel:release.wait(10))
   if sys.argv[2]=='cancel':screen.confirm_quit=lambda:False
   worker=app.request_recovery_restart(source,selector)
   if worker is not None:await worker.wait()
   assert app.is_running
   assert not getattr(app,'_recovery_restart_request',None)
   assert not app._quit_in_progress
   release.set()
  if sys.argv[2] in ('handoff','later'):
   async with asyncio.timeout(15):
    while app.is_running:await asyncio.sleep(.03)
 release.set()
 assert app.recovery_service._closed
 if sys.argv[2] in ('handoff','later'):
  request=app._recovery_restart_request
  assert type(request) is RecoveryRestart
  assert request.archive==(None if sys.argv[2]=='later' else source)
  assert request.target_config==selector
  assert request.recovery_copies==(sys.argv[2]=='later')
  assert 'not-carried-test-password' not in repr(request)
  assert password_field.value==''
 assert not blocked_attempts(),blocked_attempts()
 print('retired and reopened')
asyncio.run(main())
"""


@pytest.mark.parametrize("mode", ["handoff", "later", "busy", "cancel", "unsupported"])
def test_actual_normal_app_guarded_recovery_handoff(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.ProductionApp.test_backup_restore_composition import _ENTRY

    _run(
        tmp_path,
        "restart",
        mode,
        script=_ENTRY.split("async def main():", 1)[0] + _HANDOFF_BODY,
        timeout=70,
    )


_EXEC_DRIVER = r"""
import runpy
_guard=runpy.run_path(NETWORK_GUARD)
_guard["install"]()
blocked_attempts=_guard["blocked_attempts"]
import asyncio,os,sys
from pathlib import Path
from textual.app import App
from textual.widgets import Input,Select,Button,Static
assert 'tldw_chatbook.app' not in sys.modules
assert 'tldw_chatbook.config' not in sys.modules
assert 'TEST_PROVIDER_API_KEY' not in os.environ
source,target=map(Path,sys.argv[1:3])
copies=sys.argv[3]=="copies"
original=target.read_bytes()
def headless(app,*args,**kwargs):
 async def mounted():
  async with app.run_test(size=(100,36)) as pilot:
   screen=app.screen
   for name in ('tldw_chatbook','tldw_chatbook.Backup_Recovery.recovery_restart',
                'tldw_chatbook.Backup_Recovery.launcher',
                'tldw_chatbook.UI.Screens.backup_restore_screen'):
    relative='tldw_chatbook/__init__.py' if name=='tldw_chatbook' else name.replace('.','/')+'.py'
    assert Path(sys.modules[name].__file__).resolve()==Path(EXPECTED_INSTALLED)/relative
   if copies:
    assert screen._mode=='copies'
    assert screen.query_one('#backup-later-target',Input).value==str(target)
    assert screen._rollback_copy_id is None and screen._rollback_plan is None
    assert not screen.query_one('#backup-later-form').display
    await screen.workers.wait_for_complete()
    await pilot.pause()
    rows=list(screen.query_one('#backup-list').query(Static))
    assert len(rows)==1 and 'No local entries.' in str(rows[0].render()), ([str(row.render()) for row in rows],screen._list_delivery)
   else:
    assert screen.query_one('#backup-source',Input).value==str(source)
    assert screen.query_one('#backup-target-config',Input).value==str(target)
    assert screen.query_one('#backup-restore-mode',Select).value=='replace'
   assert screen._inspection_id is None
   assert screen.query_one('#backup-start-restore',Button).disabled
   assert app.recovery_service.current() is None
  assert app.recovery_service._closed
 asyncio.run(mounted())
 assert 'tldw_chatbook.app' not in sys.modules
 assert 'tldw_chatbook.config' not in sys.modules
 assert target.read_bytes()==original
 assert not blocked_attempts(),blocked_attempts()
 print('retired and reopened')
App.run=headless
"""


@pytest.mark.parametrize("mode", ["handoff", "later"])
def test_actual_handoff_execs_fresh_recovery_ui(tmp_path, mode, native_package):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.ProductionApp.test_backup_restore_composition import _ENTRY

    # Only the terminal driver and assertions are installed before the exact
    # production child entry. The real exec replaces the mounted normal process.
    body = _HANDOFF_BODY.replace(
        " print('retired and reopened')\nasyncio.run(main())",
        " return app._recovery_restart_request\nrequest=asyncio.run(main())\n"
        "from tldw_chatbook.Backup_Recovery import recovery_restart\n"
        "os.environ['TEST_PROVIDER_API_KEY']='private-fixture-key'\n"
        "recovery_restart._ENTRY=DRIVER+recovery_restart._ENTRY\n"
        "recovery_restart.restart(request)\n",
    )
    assert body != _HANDOFF_BODY
    expected = str(native_package.resolve())
    driver = (
        "EXPECTED_INSTALLED=" + repr(expected) + "\nNETWORK_GUARD="
        + repr(str(Path(__file__).resolve().parents[1] / "network_guard.py"))
        + "\n" + _EXEC_DRIVER
    )
    origin_check = (
        "\nimport tldw_chatbook\n"
        "assert Path(tldw_chatbook.__file__).resolve()==Path(" + repr(expected)
        + ")/'tldw_chatbook'/'__init__.py'\n"
        "assert Path(sys.modules['tldw_chatbook.app'].__file__).resolve()==Path("
        + repr(expected) + ")/'tldw_chatbook'/'app.py'\n"
    )
    _run(
        tmp_path,
        "restart",
        mode,
        script="DRIVER="
        + repr(driver)
        + "\n"
        + _ENTRY.split("async def main():", 1)[0]
        + origin_check
        + body,
        timeout=70,
        installed_package=native_package,
    )


_CLI_HANDOFF = r"""
from textual.widgets import Input,Select,Button
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Utils import terminal_utils
source=sealed(Path.home()).path
mode=sys.argv[2]
original_run=TldwCli.run
async def drive(pilot):
 app=pilot.app
 async with asyncio.timeout(20):
  while not app._ui_ready:await asyncio.sleep(.03)
 await pilot.press('f9');await pilot.pause()
 app.screen.query_one('#settings-backup-restore',Button).focus()
 await pilot.press('enter');await pilot.pause()
 await pilot.click('#backup-open-inspect')
 screen=app.screen
 screen.query_one('#backup-source',Input).value=str(source)
 await pilot.click('#backup-inspect')
 async with asyncio.timeout(15):
  while screen._inspection_id is None:await asyncio.sleep(.02)
 screen.query_one('#backup-restore-mode',Select).value='replace'
 screen.query_one('#backup-target-config',Input).value=str(selector)
 await pilot.pause()
 screen.query_one('#backup-restart',Button).focus();await pilot.press('enter')
 async with asyncio.timeout(15):
  while app.is_running:await asyncio.sleep(.03)
def headless(app,*args,**kwargs):
 return original_run(app,headless=True,size=(100,36),auto_pilot=drive)
TldwCli.run=headless
terminal_utils.warm_up_image_protocol=lambda:None
if mode=='shutdown_failure':
 original_unmount=TldwCli.on_unmount
 async def failed_unmount(app):
  await original_unmount(app)
  raise RuntimeError('fixture actual unmount failure')
 TldwCli.on_unmount=failed_unmount
calls=[]
from types import SimpleNamespace
from tldw_chatbook.Backup_Recovery import recovery_restart
recovery_restart.os=SimpleNamespace(name='posix',execve=lambda executable,args,environment:calls.append((executable,args,environment)))
sys.argv=['tldw-chatbook']
from tldw_chatbook.cli import main_cli_runner
main_cli_runner()
assert len(calls)==(mode=='success'),len(calls)
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("mode", ["success", "shutdown_failure"])
def test_actual_cli_restarts_only_after_successful_normal_run(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.ProductionApp.test_backup_restore_composition import _ENTRY

    _run(
        tmp_path,
        "restart",
        mode,
        script=_ENTRY.split("async def main():", 1)[0] + _CLI_HANDOFF,
        timeout=70,
    )


_DUPLICATE = r"""
from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart
async def main():
 app=TldwCli()
 app._recovery_restart_available=True
 entered=asyncio.Event();release=asyncio.Event()
 decisions=[]
 async def confirm():
  entered.set()
  await release.wait()
  decisions.append('accepted')
  return True
 async with app.run_test(size=(100,36)) as pilot:
  app.screen.confirm_quit=confirm
  first=app.request_recovery_restart(None,selector)
  await asyncio.wait_for(entered.wait(),5)
  app.request_recovery_restart(None,selector)
  await pilot.pause()
  release.set()
  await first.wait()
  assert decisions==['accepted'],decisions
  assert type(app._recovery_restart_request) is RecoveryRestart
 assert app.recovery_service._closed
 assert not blocked_attempts(),blocked_attempts()
 print('retired and reopened')
asyncio.run(main())
"""


def test_repeated_recovery_request_preserves_pending_quit_confirmation(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.ProductionApp.test_backup_restore_composition import _ENTRY

    _run(
        tmp_path,
        "restart",
        "duplicate",
        script=_ENTRY.split("async def main():", 1)[0] + _DUPLICATE,
        timeout=70,
    )
