"""First actual restored URL check is conservative with Dreams disabled."""
import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_CHILD = r"""
import asyncio,sys,sqlite3,json
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,_launch_descriptor
profile,control,scenario=sys.argv[1:];control=Path(control)
select_profile(profile,control)
entry=_launch_descriptor(profile,control)
from tldw_chatbook import config
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.monitoring_engine import URLMonitor,FetchBlockedError
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
from tldw_chatbook.Dreams.settings import dreams_setting
from tldw_chatbook.Utils import egress
import httpx
assert dreams_setting('enabled') is False
assert not config.get_dreams_db_path().exists()
_,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==entry.config)
activation=ActivationStore(Path(witness['store_root']))
assert not activation.allowed(witness['generation'],'db.subscriptions')
activation.approve(witness['generation'],'db.subscriptions')
db=SubscriptionsDB(config.get_subscriptions_db_path(),'restored-fetch')
row=db.get_subscription(1)
assert row['is_active'] and not row['is_paused']
options=json.loads(row['processing_options'])
expected='user_entered' if scenario=='explicit' else 'unknown'
assert options['url_provenance']==expected,options
sent=[]
async def resolve(_host):return ['192.168.1.1']
async def send(self,request,**kwargs):
 sent.append(str(request.url))
 return httpx.Response(200,request=request,headers={'content-type':'text/html'},content=b'<html><body>restored price</body></html>')
egress._resolve_async=resolve
httpx.AsyncClient.send=send
async def main():
 try:
  await URLMonitor(db).check_url(row)
 except FetchBlockedError:
  assert scenario!='explicit'
 else:
  assert scenario=='explicit','unproven restored source crossed first-fetch policy'
asyncio.run(main())
assert sent==(['https://later-private.example/pricing'] if scenario=='explicit' else [])
assert not config.get_dreams_db_path().exists(),'first fetch initialized unused Dreams storage'
assert not activation.allowed(witness['generation'],'config')
connection=db.conn
db.close()
try:connection.execute('SELECT 1')
except sqlite3.ProgrammingError:pass
else:raise AssertionError('native subscriptions handle survived before child exit')
assert not blocked_attempts(),blocked_attempts()
print('verified first restored fetch')
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
legacy=source/'subscriptions.db'
setup = "import sys,sqlite3\nfrom pathlib import Path\nlegacy=Path(sys.argv[1])\nfrom tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB\nfrom tldw_chatbook.DB.recovery_operations import recovery_adapters\nnative=SubscriptionsDB(legacy,'legacy-source')\nsource_id=native.add_subscription(name='legacy private source',type='url',source='https://later-private.example/pricing')\nassert source_id==1\nconnection=native.conn\nnative.close()\ntry:connection.execute('SELECT 1')\nexcept sqlite3.ProgrammingError:pass\nelse:raise AssertionError('fixture native handle survived explicit close')\n"
result=subprocess.run([sys.executable,"-c",setup,str(legacy)],capture_output=True,text=True,timeout=15)
assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]
from tldw_chatbook.DB.recovery_operations import recovery_adapters
stored={'missing':None,'empty':'{}','malformed':'not json','explicit':'{"url_provenance":"user_entered"}'}[sys.argv[1]]
with closing(sqlite3.connect(legacy)) as conn:
 conn.execute('UPDATE subscriptions SET processing_options=?',(stored,))
 conn.execute('DELETE FROM schema_version')
 conn.execute('INSERT INTO schema_version VALUES (2)')
 conn.commit()
 catalog=tuple(row[0] for row in conn.execute('SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name'))
 owner=next(owner for owner in recovery_adapters() if owner.owner_id=='db.subscriptions')
 assert (2,catalog) in owner.schema_policy().schema_sql,'fixture must match measured native v2 catalog'
config_payload=('[general]\nusers_name="original"\n[dreams]\nenabled=false\n[database]\nsubscriptions_db_path="'+str(legacy)+'"\n').encode()
doc=manifest(config_payload)
doc['owners']=[dict(owner_id='config',schema_version=1,capabilities=[]),dict(owner_id='db.subscriptions',schema_version=2,capabilities=[])]
doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
db_payload=legacy.read_bytes()
doc['files'].append(dict(logical_id='profile:profile:db.subscriptions',root_id='root',parent_id='root',relative_path='subscriptions.db',owner_id='db.subscriptions',payload='payload/2',size=len(db_payload),sha256=hashlib.sha256(db_payload).hexdigest()))
doc['dependency_groups'][0]['members']=['profile:profile:config','profile:profile:db.subscriptions']
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
assert 'verified first restored fetch' in result.stdout
assert legacy.read_bytes()==db_payload,'restore mutated the retained source'
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("scenario", ["missing", "empty", "malformed", "explicit"])
def test_first_restored_fetch_never_infers_user_authority(tmp_path, scenario):
    _run(tmp_path, scenario, "restored-first-fetch", script="CHILD=" + repr(_CHILD) + "\n" + _RESTORE, timeout=60)
