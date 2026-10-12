"""Native isolated restore carries Guardian history but no execution authority."""
import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_CHILD = r"""
import asyncio,sys,sqlite3
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,_launch_descriptor
profile,control,scenario=sys.argv[1:];control=Path(control)
select_profile(profile,control)
entry=_launch_descriptor(profile,control)
from tldw_chatbook import config
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian import settings
from tldw_chatbook.Guardian.check_pipeline import GuardianChecker
from tldw_chatbook.Scheduling.services.guardian_projection import GuardianProjection
from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import GuardianTrendHandler, GuardianTrendDeps, _SPAWNED_TREND_RUNS, shutdown
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
from tldw_chatbook.Dreams.profile_sources import read_guardian_topics
from tldw_chatbook.Utils.timestamps import utc_now_iso
assert config.CLI_APP_CLIENT_ID==entry.installation_id
assert settings.guardian_setting('enabled') is True
path=settings.guardian_db_path()
assert path==Path(entry.config).parent/'guardian.sqlite',path
db=GuardianDB(path)
assert db.get_rule(90)['name']=='legacy'
assert read_guardian_topics(db)==[]
_,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==entry.config)
activation=ActivationStore(Path(witness['store_root']))
assert 'db.guardian' in witness['owners']
assert not activation.allowed(witness['generation'],'db.guardian')
effects=[]
checker=GuardianChecker(db_getter=lambda:db,session_id_getter=lambda:'chat-a',notify=lambda *args:effects.append(args),append_system_row=effects.append,now=utc_now_iso)
with db.connection() as connection: before=list(connection.iterdump())
if scenario.startswith('approved'):activation.approve(witness['generation'],'db.guardian')
async def main():
 if scenario in ('check','approved'):
  await checker.check('private phrase')
  await checker.finalize_visit_async()
 elif scenario=='daily':
  handler=GuardianTrendHandler(deps_getter=lambda:GuardianTrendDeps(db,SimpleNamespace(dispatch=lambda **kwargs:effects.append(kwargs))))
  await handler.handle({'id':'guardian:trends'})
  await asyncio.gather(*tuple(_SPAWNED_TREND_RUNS))
  await shutdown()
 elif scenario in ('discovery','approved-discovery'):effects.extend(read_guardian_topics(db))
 elif scenario=='getter':
  import threading
  from tldw_chatbook.app_service_wiring import ServiceWiringMixin
  app=SimpleNamespace(guardian_db=db,_guardian_db_lock=threading.Lock())
  if ServiceWiringMixin.get_guardian_db(app) is not None:effects.append('runtime store')
 elif scenario=='projection':effects.extend(GuardianProjection(lambda:db).tasks(datetime.now(timezone.utc)))
asyncio.run(main())
with db.connection() as connection: after=list(connection.iterdump())
if scenario=='approved':assert effects and before!=after
elif scenario=='approved-discovery':assert effects and before==after
else:
 assert effects==[],effects
 assert before==after,'restored Guardian execution changed history'
assert not activation.allowed(witness['generation'],'config')
db.close()
try:connection.execute('SELECT 1')
except sqlite3.ProgrammingError:pass
else:raise AssertionError('native Guardian handle survived explicit retirement')
assert not blocked_attempts(),blocked_attempts()
print('verified isolated Guardian')
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
legacy=source/'guardian.sqlite'
with closing(sqlite3.connect(legacy)) as conn:
 conn.executescript(Path('Tests/Guardian/fixtures/guardian_v1_schema.sql').read_text())
 conn.execute("INSERT INTO guardian_rules (id,name,topic,pattern,feeds_discovery,created_at,updated_at) VALUES (90,'legacy','private','private',1,'2000-01-01','2000-01-01')")
 conn.execute("INSERT INTO guardian_alerts VALUES (90,90,'old-chat','old-visit','private','digest',?)",(datetime.now(timezone.utc).isoformat(),))
 if 'discovery' in sys.argv[1]:
  conn.executescript(Path('tldw_chatbook/DB/migrations/guardian_v1_to_v2_crisis_provenance.sql').read_text())
  conn.execute('UPDATE guardian_alerts SET crisis_provenance=0')
 conn.commit()
config_payload=('[general]\nusers_name="original"\n[guardian]\nenabled=true\n[database]\nguardian_db_path="'+str(legacy)+'"\n').encode()
doc=manifest(config_payload)
doc['owners']=[dict(owner_id='config',schema_version=1,capabilities=[]),dict(owner_id='db.guardian',schema_version=2 if 'discovery' in sys.argv[1] else 1,capabilities=[])]
doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
db_payload=legacy.read_bytes()
doc['files'].append(dict(logical_id='profile:profile:db.guardian',root_id='root',parent_id='root',relative_path='guardian.sqlite',owner_id='db.guardian',payload='payload/2',size=len(db_payload),sha256=hashlib.sha256(db_payload).hexdigest()))
doc['dependency_groups'][0]['members']=['profile:profile:config','profile:profile:db.guardian']
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
assert 'verified isolated Guardian' in result.stdout
assert legacy.read_bytes()==db_payload,'restore mutated the retained source'
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("scenario", ["check", "daily", "projection", "approved", "discovery", "approved-discovery", "getter"])
def test_restored_guardian_requires_current_owner_activation(tmp_path, scenario):
    _run(tmp_path, scenario, "guardian-activation", script="CHILD=" + repr(_CHILD) + "\n" + _RESTORE, timeout=60)
