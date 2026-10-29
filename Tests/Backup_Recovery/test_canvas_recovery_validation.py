"""Current Canvas payload constraints survive bounded, inert recovery reads."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import hashlib,sqlite3,sys
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Canvas.repository import CanvasRepository
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate,_check,_current_restrictions
from tldw_chatbook.DB.private_sqlite import open_recovery_validation
stage=Path.home()/'stage';stage.mkdir(mode=0o700)
path=stage/'core.db'
if sys.argv[1]=='hybrid':
 from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
 subscriptions=SubscriptionsDB(path);subscriptions.close()
db=CharactersRAGDB(path,'canvas-recovery')
conversation=db.add_conversation({'title':'Retained Canvas'})
message=db.add_message({'conversation_id':conversation,'sender':'assistant','role':'assistant','content':'Canvas origin'})
created=CanvasRepository(db).create_canvas(conversation,title='Retained payload',source='<main>recovery-canvas-source</main>',runtime_profile='canvas-v1',actor_kind='assistant',origin_message_id=message,origin_turn_id='original-turn')
row=tuple(db.get_connection().execute('SELECT * FROM canvas_revisions').fetchone())
db.close()
if sys.argv[1]=='hybrid':
 from tldw_chatbook.DB.recovery_operations import recovery_adapters
 owner=next(a for a in recovery_adapters() if a.owner_id=='db.subscriptions')
else:owner=next(a for a in core_adapters() if a.owner_id=='db.chachanotes.primary')
case=sys.argv[2]
if case=='corrupt':
 data=path.read_bytes();original=b'<main>recovery-canvas-source</main>'
 assert data.count(original)>=1
 path.write_bytes(data.replace(original,b'<main>recovery-canvas-BROKEN</main>'))
 assert owner.validate(path)==('invalid_sqlite_integrity',),owner.validate(path)
 assert validate_candidate(owner,path,Event(),migrate=False)==('invalid_sqlite_integrity',)
elif case=='custom':
 with sqlite3.connect(path) as connection:
  connection.execute('CREATE VIEW untrusted_canvas AS SELECT canvas_revision_payload_valid(x,y,z) FROM imported_payload')
 before=path.read_bytes()
 assert owner.validate(path)==('unsupported_schema',)
 assert validate_candidate(owner,path,Event(),migrate=True)==('unsupported_schema',)
 assert path.read_bytes()==before
else:
 assert owner.validate(path)==(),owner.validate(path)
 assert validate_candidate(owner,path,Event(),migrate=False)==()
 with open_recovery_validation(owner.owner_id,path,writable=False,with_restrictions=True) as (connection, restrictions):
  assert _check(connection,owner,owner.schema_policy(),restrictions)[0]==()
  assert connection.execute('PRAGMA trusted_schema').fetchone()==(0,)
  for sql in ('PRAGMA trusted_schema=ON','DELETE FROM canvas_revisions',"SELECT canvas_revision_payload_valid(x'00','bad',1)"):
   try:connection.execute(sql)
   except sqlite3.DatabaseError:pass
   else:raise AssertionError('validation retained schema or write authority')
  assert _current_restrictions(connection) is restrictions
  steps,deadline=restrictions.steps,restrictions.deadline
  with open_recovery_validation(owner.owner_id,path,writable=False,with_restrictions=True) as (nested,nested_restrictions):
   assert _current_restrictions(nested) is nested_restrictions
   assert _current_restrictions(connection) is None
  assert _current_restrictions(connection) is restrictions
  assert (restrictions.steps,restrictions.deadline)==(steps,deadline)
  assert restrictions.steps>0
  restrictions.deadline=-1
  try:_check(connection,owner,owner.schema_policy(),restrictions)
  except ValueError as error:assert error.args==('sqlite_resource_limit',)
  else:raise AssertionError('candidate validation reset its original catalog budget')
 assert _current_restrictions(connection) is None
 if case=='rewrite':
  from tldw_chatbook.Backup_Recovery.credentials import _rewrite_database
  _rewrite_database(stage,path,owner.owner_id)
  assert owner.validate(path)==()
  assert validate_candidate(owner,path,Event(),migrate=False)==()
 with sqlite3.connect(path) as connection:
  assert tuple(connection.execute('SELECT * FROM canvas_revisions').fetchone())==row
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("case", ["valid", "corrupt", "custom", "rewrite"])
@pytest.mark.parametrize("kind", ["core", "hybrid"])
def test_current_canvas_schema_is_checked_without_imported_authority(
    tmp_path, kind, case
):
    _run(tmp_path, kind, case, script=_SCRIPT)


_V81_SCRIPT = (
    _SCRIPT.replace(
        "db=CharactersRAGDB(path,'canvas-recovery')",
        "from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version\n"
        "historical=chachanotes_db_at_version(path,81)\n"
        "db=historical.__enter__()",
    )
    .replace(
        "db.close()",
        "historical.__exit__(None,None,None)",
    )
    .replace(
        "case=sys.argv[2]",
        "assert owner.validate(path)==(),owner.validate(path)\n"
        "if sys.argv[1]=='core':\n"
        " assert validate_candidate(owner,path,Event(),migrate=True)==()\n"
        "else:\n"
        " migrated=CharactersRAGDB(path,'hybrid-upgrade');migrated.close()\n"
        "with sqlite3.connect(path) as connection:\n"
        " assert connection.execute('SELECT version FROM db_schema_version').fetchone()==(max(owner.schema_policy().versions),)\n"
        "case=sys.argv[2]",
    )
)


@pytest.mark.parametrize("kind", ["core", "hybrid"])
def test_genuine_v81_recovery_preserves_payload_and_restricted_authority(
    tmp_path, kind
):
    _run(tmp_path, kind, "valid", script=_V81_SCRIPT)


_LINEAGE_SCRIPT = r"""
import sqlite3
from pathlib import Path
from threading import Event
from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V81_SCHEMAS
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.DB.canvas_payload_validation import install_canvas_revision_payload_validator
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
owner=next(a for a in core_adapters() if a.owner_id=='db.chachanotes.primary')
assert len(CHACHANOTES_V81_SCHEMAS)==4
for index,schema in enumerate(CHACHANOTES_V81_SCHEMAS):
 path=Path.home()/f'lineage-{index}.sqlite'
 with sqlite3.connect(path) as connection:
  install_canvas_revision_payload_validator(connection)
  tables=[sql for sql in schema if sql.startswith(('CREATE TABLE','CREATE VIRTUAL TABLE'))]
  for sql in tables+[sql for sql in schema if sql not in tables]:
   if connection.execute('SELECT 1 FROM sqlite_schema WHERE sql=?',(sql,)).fetchone() is None:
    connection.execute(sql)
  connection.execute("INSERT INTO db_schema_version VALUES ('rag_char_chat_schema',81)")
 assert owner.validate(path)==(),(index,owner.validate(path))
 assert validate_candidate(owner,path,Event(),migrate=True)==()
 with sqlite3.connect(path) as connection:
  assert connection.execute('SELECT version FROM db_schema_version').fetchone()==(max(owner.schema_policy().versions),)
  assert connection.execute("SELECT [unique] FROM pragma_index_list('canvas_revisions') WHERE name='idx_canvas_revisions_canvas_sequence'").fetchone()==(0,)
print('retired and reopened')
"""


def test_all_four_frozen_v81_lineages_remain_accepted_and_migrate(tmp_path):
    _run(tmp_path, "core", "lineages", script=_LINEAGE_SCRIPT)
