"""Observe foreign overlap around the original frozen actual8; no signals."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
E=Path(__file__).resolve().parent
G=Path(os.environ['FIRSTSEND_SOURCE'])
PY=sys.executable

def processes(owned=()):
 rows={}
 for line in subprocess.check_output(['ps','-axo','pid=,ppid=,comm=,command='],text=True).splitlines():
  f=line.split(None,3)
  if len(f)==4:rows[int(f[0])]=(int(f[1]),f[2],f[3])
 excluded={os.getpid(),*owned}
 for _ in range(20):
  children={pid for pid,row in rows.items() if row[0] in excluded}
  if children<=excluded:break
  excluded|=children
 output=[]
 for pid,(parent,comm,command) in rows.items():
  if pid in excluded:continue
  kinds=[]
  if 'pytest' in command:kinds.append('pytest')
  if any(token in command for token in ('pr3023-','pr3010-native-deepseek-final-launch.py','textual_serve','textual-serve','cdp-driver-preparation')):kinds.append('benchmark_app')
  if kinds:output.append(dict(pid=pid,ppid=parent,categories=kinds))
 return output

manifest=json.loads((E/'postsave-current31-rag-path-paired-frozen-chain-r1.json').read_text())
seal=json.loads((E/'postsave-current31-rag-path-prelaunch-seal-r1.json').read_text())
quiet_path=E/'postsave-current31-rag-path-quiet-launch-r1.json'
quiet=json.loads(quiet_path.read_text())
assert quiet['continuous_quiet_30s']
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(quiet['checked_utc'])).total_seconds()<30
for name,digest in seal['files'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest,name
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==manifest['candidate_head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
python=json.loads((E/'postsave-current31-rag-path-python-prelaunch-r1.json').read_text())
names=subprocess.check_output(['git','ls-files','-z'],cwd=G).decode().split('\0')
current={n:hashlib.sha256((G/n).read_bytes()).hexdigest() for n in names if n.endswith('.py') and (G/n).is_file()}
assert current==python, 'whole Python source freshness changed before launch'
assert not processes()
receipt=E/'postsave-current31-rag-path-host-watch-r1.json'
assert not receipt.exists() and not(E/manifest['packet']).exists()
observations=[];overlaps=[];start=time.monotonic()
with (E/'postsave-current31-rag-path-measure-r1.log').open('x') as handle:
 run=subprocess.Popen([PY,str(E/'pr3023-postsave-current31-rag-path-firstsend-measure-r1.py')],cwd=G,stdout=handle,stderr=subprocess.STDOUT)
 while True:
  blockers=processes((run.pid,))
  row=dict(elapsed_s=round(time.monotonic()-start,3),blockers=blockers)
  observations.append(row)
  if blockers:overlaps.append(row)
  status=run.poll()
  if status is not None:break
  time.sleep(2)
result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),measure_exit=status,observations=observations,overlaps=overlaps,metadata_poll_seconds=2,limitations='Observed overlap metadata only; no global absence or complete process census claim. All original per-run source/physical-owned-PID/port gates remain mandatory.')
with receipt.open('x') as handle:json.dump(result,handle,indent=2)
print(json.dumps(dict(measure_exit=status,overlap_observations=len(overlaps),receipt=receipt.name)))
assert status==0 and not overlaps, 'original actual8 or quiet-host qualification failed; do not compute acceptance statistics'
