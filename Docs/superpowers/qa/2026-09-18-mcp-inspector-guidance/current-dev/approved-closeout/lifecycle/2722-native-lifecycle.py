import hashlib,json,sqlite3,subprocess
from pathlib import Path
import portalocker
root=Path('<tmp>/tldw-32836-approved-001');repo=Path.cwd();e=json.loads((root/'evidence/result.json').read_text());pid=e['pid']
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((root/'default-before.json').read_text());after={p:hashfile(Path(p)) if Path(p).exists() else None for p in before};r={'pid':pid,'pid_absent':subprocess.run(['ps','-p',str(pid)],capture_output=True).returncode==1,'exit_code':int((root/'exit-code').read_text()),'passed':e['passed'],'app_run_returned':e['app_run_returned'],'app_return_code':e['app_return_code'],'app_exception':e['app_exception'],'defaults_unchanged':before==after,'database_checks':{},'counts':{},'network_attempts':e['network_attempts']}
for p in sorted((root/'data').rglob('*.db')):
 with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:
  r['database_checks'][str(p.relative_to(root))]=db.execute('pragma quick_check').fetchall()
  if p.name=='chachanotes.db':
   for table in ['conversations','messages']:r['counts'][table]=db.execute('select count(*) from '+table).fetchone()[0]
locks=list(root.rglob('.instance.lock'));assert len(locks)==1
with locks[0].open('a') as lock:
 portalocker.lock(lock,portalocker.LockFlags.EXCLUSIVE|portalocker.LockFlags.NON_BLOCKING);r['instance_lock_reacquired']=True;portalocker.unlock(lock)
r['source_hashes_match']=all(hashfile(repo/p)==h for p,h in e['source_hashes'].items());r['runner_hash_matches']=hashfile(repo/'Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/native_check.py')==e['runner_sha256'];r['sentinels_unchanged']=all(hashfile(Path(p))==h for p,h in json.loads((root/'sentinel-before.json').read_text()).items())
r['faulthandler_bytes']={str(p.relative_to(root)):p.stat().st_size for p in root.rglob('*faulthandler*') if p.is_file()};r['errors']=[{'path':str(p.relative_to(root)),'line':line} for p in root.rglob('*.log') for line in p.read_text(errors='replace').splitlines() if any(k in line for k in ['| ERROR','| CRITICAL','unhandled_exception','Traceback (most recent call last)','WorkerError','EmptyStack'])]
(root/'lifecycle.json').write_text(json.dumps(r,indent=2)+'\n')
assert all(r[k] for k in ['pid_absent','passed','app_run_returned','defaults_unchanged','instance_lock_reacquired','source_hashes_match','runner_hash_matches','sentinels_unchanged'])
assert r['exit_code']==0 and r['app_return_code']==0 and r['app_exception'] is None
assert len(r['database_checks'])>=10 and all(c==[('ok',)] for c in r['database_checks'].values()) and r['counts']=={'conversations':0,'messages':0}
assert not r['errors'] and not r['network_attempts'] and r['faulthandler_bytes'] and not any(r['faulthandler_bytes'].values())
print('Native lifecycle and all captured source hashes verified')
