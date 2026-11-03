import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
phase=sys.argv[1]
plan=json.loads((root/'planned-commands.json').read_text())[phase]
results=[]
for index,argv in enumerate(plan):
 log=root/f'{phase}-{index}.log'
 started=time.monotonic()
 with log.open('w') as output: rc=subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT).returncode
 results.append({'argv':argv,'cwd':str(Path.cwd()),'exit_code':rc,'seconds':round(time.monotonic()-started,3),'log':str(log)})
 print(phase,index,'exit',rc,flush=True)
(root/f'{phase}-commands-results.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(any(row['exit_code'] for row in results))
