import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).parent
name=sys.argv[1]
argv=json.loads((p/'commands.json').read_text())[name]
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
with (p/(name+'.log')).open('w') as out:
 r=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env)
(p/(name+'-result.json')).write_text(json.dumps({'argv':argv,'returncode':r.returncode})+'\n')
print(name,r.returncode)
raise SystemExit(r.returncode)
