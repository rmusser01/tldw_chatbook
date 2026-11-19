"""Actual isolated Dreams restoration preserves history without runtime consent."""
import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_CHILD = r"""
import asyncio,sys,sqlite3,threading
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,_launch_descriptor
profile,control,scenario=sys.argv[1:];control=Path(control)
select_profile(profile,control)
entry=_launch_descriptor(profile,control)
from tldw_chatbook import config
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.settings import dreams_setting
from tldw_chatbook.Dreams.cycle_service import CycleDeps,run_cycle
from tldw_chatbook.Dreams.track_service import run_track_check
from tldw_chatbook.Dreams.track_reconcile import reconcile_at_boot
from tldw_chatbook.Scheduling.services.dreams_projection import DreamsProjection
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
assert config.CLI_APP_CLIENT_ID==entry.installation_id
assert dreams_setting('enabled') is True
path=config.get_dreams_db_path()
assert path==Path(entry.config).parent/'dreams.sqlite',path
db=DreamsDB(path)
_,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==entry.config)
activation=ActivationStore(Path(witness['store_root']))
assert 'db.dreams' in witness['owners']
assert not activation.allowed(witness['generation'],'db.dreams')
effects=[]
def forbidden(*args,**kwargs):
 effects.append('external callback')
 return []
deps=CycleDeps(db,lambda:None,lambda:None,lambda:None,lambda:None,lambda:forbidden,perform_search=forbidden)
with db.connection() as connection:before=list(connection.iterdump())
if scenario.startswith('approved'):activation.approve(witness['generation'],'db.dreams')
async def main():
 if scenario=='cycle':await run_cycle(deps,trigger='manual')
 elif scenario=='track':await run_track_check(deps,71)
 elif scenario in ('projection','approved-projection'):
  effects.extend(DreamsProjection(lambda:db).tasks(datetime.now(timezone.utc)))
 elif scenario=='getter':
  from tldw_chatbook.app_service_wiring import ServiceWiringMixin
  app=SimpleNamespace(dreams_db=db,_dreams_db_lock=threading.Lock())
  if ServiceWiringMixin.get_dreams_db(app) is not None:effects.append('runtime store')
 elif scenario=='boot':
  with db.transaction() as connection:
   connection.execute("UPDATE dream_tracked_items SET suspension=?",('{"settled":"off"}',))
  with db.connection() as connection:before[:]=list(connection.iterdump())
  app=SimpleNamespace(get_dreams_db=lambda:db,local_watchlists_service=None)
  await reconcile_at_boot(app)
asyncio.run(main())
with db.connection() as connection:after=list(connection.iterdump())
if scenario=='approved-projection':assert effects and before==after
else:
 assert effects==[],effects
 assert before==after,'unapproved restored Dreams execution changed history'
assert not activation.allowed(witness['generation'],'config')
db.close_all()
try:connection.execute('SELECT 1')
except sqlite3.ProgrammingError:pass
else:raise AssertionError('native Dreams handle survived explicit retirement')
assert not blocked_attempts(),blocked_attempts()
print('verified isolated Dreams')
"""

_RESTORE = r"""
import os,sys,json,hashlib,sqlite3,zipfile,subprocess
from contextlib import closing
from pathlib import Path
from datetime import datetime,timezone
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
base=Path.home()
source=base/'source';source.mkdir(mode=0o700)
legacy=source/'dreams.sqlite'
from Tests.Dreams.test_schema_lineage import historical_store,LAYOUTS
historical_store(legacy,next(row for row in LAYOUTS if row['version']==6))
with closing(sqlite3.connect(legacy)) as conn:
 conn.execute("UPDATE dream_tracked_items SET mechanism='question',query_template='safe query',subscription_id=NULL")
 conn.commit()
config_payload=('[general]\nusers_name="original"\n[dreams]\nenabled=true\n[database]\ndreams_db_path="'+str(legacy)+'"\n').encode()
doc=manifest(config_payload)
doc['owners']=[dict(owner_id='config',schema_version=1,capabilities=[]),dict(owner_id='db.dreams',schema_version=6,capabilities=[])]
doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
db_payload=legacy.read_bytes()
doc['files'].append(dict(logical_id='profile:profile:db.dreams',root_id='root',parent_id='root',relative_path='dreams.sqlite',owner_id='db.dreams',payload='payload/2',size=len(db_payload),sha256=hashlib.sha256(db_payload).hexdigest()))
doc['dependency_groups'][0]['members']=['profile:profile:config','profile:profile:db.dreams']
zip_path=source/'fixture.zip'
with zipfile.ZipFile(zip_path,'w') as archive:
 archive.writestr('manifest.json',json.dumps(doc))
 archive.writestr('payload/1',config_payload)
 archive.writestr('payload/2',db_payload)
archive=acquire(zip_path,source/'acquired',ArchiveLimits(),None,Event())
dest=base/'destinations';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
control=base/'isolated-control'
profile=restore_isolated(archive,plan,control,Event())
result=subprocess.run([sys.executable,'-c',CHILD,profile,str(control),sys.argv[1]],capture_output=True,text=True,timeout=25)
assert result.returncode==0,result.stdout[-1000:]+result.stderr[-6000:]
assert 'verified isolated Dreams' in result.stdout
assert legacy.read_bytes()==db_payload,'restore mutated the retained source'
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""

@pytest.mark.parametrize("scenario", ["cycle", "track", "projection", "getter", "boot", "approved-projection"])
def test_restored_dreams_requires_current_owner_activation(tmp_path, scenario):
    _run(tmp_path, scenario, "dreams-activation", script="CHILD=" + repr(_CHILD) + "\n" + _RESTORE, timeout=60)
