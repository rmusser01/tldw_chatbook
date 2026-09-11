"""Real terminal suspension retains a finite service wait without repaint deadlock."""

import errno
import json
import os
import select
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import pytest

_CHILD = r"""
import asyncio,faulthandler,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
root=Path.home().parent
mode=sys.argv[1]
trace=(root/'stacks.log').open('w')
faulthandler.dump_traceback_later(8,file=trace)
from rich.text import Text
from textual.app import App
from textual.widgets import Static
from textual.worker import WorkerCancelled
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
entered,release,completed=threading.Event(),threading.Event(),threading.Event()

class FiniteService(RecoveryService):
 def start_open_profile(self,profile):
  assert profile=='terminal-fixture'
  entered.set()
  if mode=='start_error':raise ValueError('recovery_service_closed')
  def finish(operation,cancel):
   try:
    assert release.wait(5),'fixture release not reached'
    if mode=='failed_result':raise ValueError('fixture service failure')
    self._update(operation,phase='fixture_finished')
   finally:completed.set()
  return self._start('open_profile',finish)

service=FiniteService(root/'control')

class Host(App):
 # Exercise the actual production method with a bounded service boundary.
 # This test does not claim to launch or qualify a restored profile.
 open_recovery_profile=TldwCli.open_recovery_profile
 recovery_service=service
 def compose(self):yield Static('Parent terminal')

async def main():
 app=Host()
 try:
  async with app.run_test(headless=False,size=(80,25)):
   assert app._driver.can_suspend and not app.is_headless
   worker=app.open_recovery_profile('terminal-fixture')
   async with asyncio.timeout(3):
    while not entered.is_set():await asyncio.sleep(.01)
   if mode=='start_error':
    await worker.wait()
   else:
    stopped=app._driver._writer_thread
    assert not stopped.is_alive()
    queued=stopped._queue.qsize()
    (root/'suspended.json').write_text(json.dumps({'writer_stopped':True,'queued_before':queued}))
    # Real display calls exceed the stopped writer queue capacity without a batch.
    for index in range(40):app._display(app.screen,Text('Repaint '+str(index)))
    assert stopped._queue.qsize()==queued
    assert not completed.is_set()
    if mode=='cancel':
     worker.cancel()
     await asyncio.sleep(0)
     await asyncio.sleep(0)
     assert not completed.is_set() and not stopped.is_alive()
    release.set()
    try:
     async with asyncio.timeout(3):await worker.wait()
    except WorkerCancelled:assert mode=='cancel'
    else:assert mode!='cancel'
    assert completed.is_set()
    assert service.current()['state']==('failed' if mode=='failed_result' else 'succeeded')
   evidence={'mode':mode,'writer_resumed':app._driver._writer_thread.is_alive(),'batch_count':app._batch_count,'service_completed':completed.is_set()}
   (root/'result.json').write_text(json.dumps(evidence))
   assert evidence['writer_resumed'] and evidence['batch_count']==0,evidence
   app._display(app.screen,Text('Parent resumed'))
   assert not blocked_attempts(),blocked_attempts()
 finally:
  release.set()
  await asyncio.to_thread(service.close)
  faulthandler.cancel_dump_traceback_later()
  trace.close()
asyncio.run(main())
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="Actual native terminal driver")
@pytest.mark.parametrize("mode", ["success", "cancel", "failed_result", "start_error"])
def test_parent_terminal_resumes_after_service_wait(tmp_path, mode):
    import pty

    for name in ("home", "config", "data", "cache", "tmp"):
        (tmp_path / name).mkdir(mode=0o700)
    selector = tmp_path / "config" / "config.toml"
    selector.write_text('[general]\nusers_name="terminal-fixture"\n')
    selector.chmod(0o600)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
    }
    environment.update(
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        TMPDIR=str(tmp_path / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        TERM="xterm-256color",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    master, slave = pty.openpty()
    process = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", _CHILD, mode],
        cwd=environment["PYTHONPATH"],
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    deadline = time.monotonic() + 15
    try:
        with (tmp_path / "terminal.log").open("wb") as output:
            while True:
                assert time.monotonic() < deadline, "Terminal child timed out"
                readable, _, _ = select.select([master], [], [], 0.05)
                if readable:
                    try:
                        data = os.read(master, 65536)
                    except OSError as error:
                        if error.errno != errno.EIO:
                            raise
                        break
                    if not data:
                        break
                    output.write(data)
                elif process.poll() is not None:
                    break
        assert process.wait(timeout=3) == 0, (tmp_path / "terminal.log").read_text(
            errors="replace"
        )[-5000:]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)
    evidence = json.loads((tmp_path / "result.json").read_text())
    assert evidence["writer_resumed"] and evidence["batch_count"] == 0
