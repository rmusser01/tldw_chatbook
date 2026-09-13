"""Native process exits preserve the recovered-media owner's deletion journal."""

import json
import os
import subprocess  # nosec B404 - isolated native crash children, fixed programs only
import sys
from pathlib import Path

import pytest

_PRIVATE = r"""
import hashlib,json,os,sqlite3,sys
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia
fixture=Path(os.environ['RECOVERED_CRASH_FIXTURE'])
root=fixture/'data'/'recovered'
def snapshot():
 with closing(sqlite3.connect((root/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True)) as connection:
  return {
   'assets':connection.execute('SELECT asset_id,digest,size,media_type,state FROM assets ORDER BY asset_id').fetchall(),
   'refs':connection.execute('SELECT profile,message,slug,media_type,asset_id FROM refs ORDER BY profile,message,slug,media_type').fetchall(),
   'tombstones':connection.execute('SELECT asset_id,version,references_json FROM tombstones ORDER BY asset_id').fetchall(),
   'operations':connection.execute('SELECT asset_id,kind FROM operations ORDER BY asset_id').fetchall(),
  }
def record(name,value):
 with (fixture/name).open('x',encoding='utf-8') as output:
  json.dump(value,output,sort_keys=True);output.flush();os.fsync(output.fileno())
"""

_CRASH = (
    _PRIVATE
    + r"""
boundary=sys.argv[1]
source=fixture/'source';source.write_bytes(b'deleted opaque recovered media');source.chmod(0o600)
survivor_source=fixture/'survivor-source';survivor_source.write_bytes(b'unrelated retained media');survivor_source.chmod(0o600)
store=RecoveredMedia(root)
references=[dict(profile='profile-a',message=message,slug='image',media_type='image/png') for message in ('first','second')]
asset=store.retain(source,**references[0]);store.add_reference(asset,**references[1])
other_reference=dict(profile='profile-b',message='keep',slug='video',media_type='video/webm')
other=store.retain(survivor_source,**other_reference)
target=store.resolve(asset)[1];survivor=store.resolve(other)[1]
before=snapshot()
assert before['operations']==[] and before['tombstones']==[]
record('before.json',dict(catalog=before,asset=asset,other=other,references=references,other_reference=other_reference,target=str(target),survivor=str(survivor)))
def crash():
 observed=snapshot()  # A separate read-only connection observes committed state.
 assert observed['operations']==[(asset,'delete')]
 assert next(row[4] for row in observed['assets'] if row[0]==asset)=='deleted'
 assert observed['refs']==before['refs']
 assert len(observed['tombstones'])==1
 assert json.loads(observed['tombstones'][0][2])==[list(row[:4]) for row in before['refs'] if row[4]==asset]
 assert target.exists()==(boundary=='committed')
 assert survivor.read_bytes()==survivor_source.read_bytes()
 assert not blocked_attempts()
 record('boundary.json',dict(boundary=boundary,catalog=observed,target_exists=target.exists(),survivor_digest=hashlib.sha256(survivor.read_bytes()).hexdigest()))
 os._exit(91 if boundary=='committed' else 92)
if boundary=='committed':
 def at_finish(connection,asset_id):
  assert asset_id==asset and not connection.in_transaction
  crash()
 store._finish_delete=at_finish
else:
 original_unlink=os.unlink
 parent_identity=(root.stat().st_dev,root.stat().st_ino)
 def after_unlink(path,*,dir_fd=None):
  exact=dir_fd is not None and path==target.name and (os.fstat(dir_fd).st_dev,os.fstat(dir_fd).st_ino)==parent_identity
  if exact:assert target.exists()
  result=original_unlink(path,dir_fd=dir_fd)
  if exact:crash()
  return result
 os.unlink=after_unlink
store.delete(asset)
raise AssertionError('native crash boundary was not reached')
"""
)

_RECOVER = (
    _PRIVATE
    + r"""
before=json.loads((fixture/'before.json').read_text())
boundary=json.loads((fixture/'boundary.json').read_text())
assert json.loads(json.dumps(snapshot()))==boundary['catalog']
target=Path(before['target']);survivor=Path(before['survivor'])
assert target.exists()==boundary['target_exists']
store=RecoveredMedia(root)  # Its real constructor finishes pending owner operations.
assert store.resolve(before['asset'])==('deleted',None)
for reference in before['references']:
 assert store.resolve_reference(**reference)==('deleted',None)
assert store.resolve_reference(**before['other_reference'])==('ready',survivor)
assert hashlib.sha256(survivor.read_bytes()).hexdigest()==boundary['survivor_digest']
assert not target.exists()
after=json.loads(json.dumps(snapshot()))
assert after['operations']==[]
assert after['assets']==boundary['catalog']['assets']
assert after['refs']==before['catalog']['refs']
assert after['tombstones']==boundary['catalog']['tombstones']
store.recover()  # Completed recovery is idempotent through the same real owner.
assert json.loads(json.dumps(snapshot()))==after
assert not blocked_attempts()
record('recovered.json',dict(catalog=after,target_exists=False,survivor_digest=hashlib.sha256(survivor.read_bytes()).hexdigest()))
print('RECOVERED_DELETION_AND_REFERENCES_PRESERVED',flush=True)
"""
)


@pytest.mark.parametrize("boundary", ["committed", "unlinked"])
def test_recovered_media_delete_survives_native_process_exit(tmp_path, boundary):
    root = tmp_path.resolve()
    for name in ("home", "config", "data", "cache", "tmp"):
        (root / name).mkdir(mode=0o700)
    selector = root / "config" / "config.toml"
    selector.write_text(
        '[general]\nusers_name="test"\n[paths]\ndata_dir='
        + json.dumps(str(root / "data"))
        + "\n",
        encoding="utf-8",
    )
    selector.chmod(0o600)
    environment = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
        if name in os.environ
    }
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        RECOVERED_CRASH_FIXTURE=str(root),
    )
    for name, script, expected in (
        ("crash", _CRASH, 91 if boundary == "committed" else 92),
        ("recover", _RECOVER, 0),
    ):
        log = root / (name + ".log")
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(  # nosec B603 - fixed Python programs, private fixture paths, no shell
                [sys.executable, "-c", script, boundary],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        assert result.returncode == expected, log.read_text()[-8000:]
    observed = json.loads((root / "boundary.json").read_text())
    recovered = json.loads((root / "recovered.json").read_text())
    assert observed["target_exists"] == (boundary == "committed")
    assert observed["catalog"]["operations"]
    assert recovered["catalog"]["operations"] == []
    assert recovered["catalog"]["refs"] == observed["catalog"]["refs"]
    assert recovered["catalog"]["tombstones"] == observed["catalog"]["tombstones"]
    assert recovered["survivor_digest"] == observed["survivor_digest"]
