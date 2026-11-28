"""Metadata-only continuous readiness; never starts or interrupts an app."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
E=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
args=parser.parse_args()
output=E/f'postsave-current31-rag-path-quiet-{args.label}-r1.json'
assert not output.exists()
start=time.monotonic();quiet_since=None;observations=[]
while True:
 rows=[]
 for line in subprocess.check_output(['ps','-axo','pid=,ppid=,etime=,command='],text=True).splitlines():
  fields=line.split(None,3)
  if len(fields)<4:continue
  pid,parent,elapsed,command=fields
  if int(pid)==os.getpid():continue
  categories=[name for name,hit in (
   ('pytest','pytest' in command),
   ('benchmark_app',any(text in command for text in ('pr3023-','pr3010-native-deepseek-final-launch.py','textual_serve','textual-serve','cdp-driver-preparation'))),
  ) if hit]
  if categories:rows.append(dict(pid=int(pid),ppid=int(parent),elapsed=elapsed,categories=categories))
 now=time.monotonic()
 observations.append(dict(elapsed_s=round(now-start,3),blockers=rows))
 if rows:quiet_since=None
 elif quiet_since is None:quiet_since=now
 passed=quiet_since is not None and now-quiet_since>=30
 if passed or now-start>=55:
  receipt=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),continuous_quiet_30s=passed,observations=observations,no_application_launched=True,no_other_work_interrupted=True)
  with output.open('x') as f:json.dump(receipt,f,indent=2)
  print(json.dumps(dict(continuous_quiet_30s=passed,last_blockers=rows,receipt=output.name)))
  raise SystemExit(0 if passed else 42)
 time.sleep(2)
