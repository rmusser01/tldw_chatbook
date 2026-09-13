"""Exact installed dictionary-update DDL is a native core schema variant."""

import json
import os
import subprocess  # nosec B404 - fixed private native fixture
import sys
from pathlib import Path

import pytest

_PROGRAM = r"""
import sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery.chat_source_participants import build_dictionary_service
from tldw_chatbook.DB.recovery_core import core_adapters
mode=sys.argv[1]
db=config.get_chachanotes_db_lazy()
core=next(a for a in core_adapters() if a.owner_id=='db.chachanotes.primary')
try:
 service=build_dictionary_service(db)
 record=service.create_dictionary({'name':'Native schema','description':'Retained dictionary'})
 assert core.validate(db.db_path)==()
 if mode!='initial':
  service.add_entry(record['id'],{'pattern':'retained','replacement':'native value'})
  assert service.list_entries(record['id'])['entries'][0]['replacement']=='native value'
  connection=db.get_connection()
  trigger=connection.execute("SELECT sql FROM sqlite_schema WHERE type='trigger' AND name='chat_dictionaries_au'").fetchone()[0]
  if mode in ('body','predicate'):
   changed=trigger.replace('NEW.name','OLD.name') if mode=='body' else trigger.replace('WHERE NEW.deleted = 0','WHERE NEW.deleted = 1')
   assert changed!=trigger
   connection.execute('DROP TRIGGER chat_dictionaries_au')
   connection.execute(changed)
  elif mode=='missing':connection.execute('DROP TRIGGER chat_dictionaries_au')
  elif mode=='extra':connection.execute('CREATE TRIGGER unexpected AFTER UPDATE ON chat_dictionaries BEGIN SELECT 1; END')
  elif mode=='unrelated':
   other=connection.execute("SELECT sql FROM sqlite_schema WHERE type='trigger' AND name='chat_dictionaries_ai'").fetchone()[0]
   changed=other.replace('AFTER INSERT','AFTER  INSERT')
   assert changed!=other
   connection.execute('DROP TRIGGER chat_dictionaries_ai')
   connection.execute(changed)
  elif mode=='version':connection.execute("UPDATE db_schema_version SET version=999 WHERE schema_name='rag_char_chat_schema'")
  else:assert mode=='native_update'
  connection.commit()
 expected=() if mode in ('initial','native_update') else ('unsupported_schema_version',) if mode=='version' else ('unsupported_schema',)
 assert core.validate(db.db_path)==expected,(mode,core.validate(db.db_path))
 from threading import Event
 from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
 # Restricted validation reads an immutable disposable candidate, never live WAL.
 db.close()
 source=Path(db.db_path)
 assert not Path(str(source)+'-wal').exists() or Path(str(source)+'-wal').stat().st_size==0
 candidate=Path.home()/'dictionary-candidate.db'
 candidate.write_bytes(source.read_bytes())
 candidate.chmod(0o600)
 issues=validate_candidate(core,candidate,Event(),migrate=False)
 assert issues==expected,(mode,'restricted candidate validation',issues,expected)
 assert not blocked_attempts(),blocked_attempts()
 print('exact native schema checked',mode,flush=True)
finally:db.close()
"""


@pytest.mark.parametrize(
    "mode",
    [
        "initial",
        "native_update",
        "body",
        "predicate",
        "extra",
        "missing",
        "unrelated",
        "version",
    ],
)
def test_native_dictionary_schema_is_exact(tmp_path, mode):
    for name in ("home", "config", "data", "cache", "tmp"):
        (tmp_path / name).mkdir(mode=0o700)
    selector = tmp_path / "config" / "config.toml"
    selector.write_text(
        '[general]\nusers_name="test"\n[paths]\ndata_dir='
        + json.dumps(str(tmp_path / "data"))
        + "\n"
    )
    selector.chmod(0o600)
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "TMPDIR": str(tmp_path / "tmp"),
        "TLDW_CONFIG_PATH": str(selector),
        "TLDW_TEST_MODE": "1",
        "TLDW_DISABLE_CONFIG_WATCH": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "PYTHONPATH": str(Path.cwd()),
    }
    log = tmp_path / "child.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", _PROGRAM, mode],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )  # nosec B603 - fixed source and private synthetic selected config
    assert result.returncode == 0, log.read_text()[-10000:]
