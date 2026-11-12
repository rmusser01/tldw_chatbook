import os,subprocess,json,time,sys
from pathlib import Path
p=Path(__file__).parent;label=sys.argv[1]
argv=['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','pytest',*sys.argv[2:],'-q','-p','no:cacheprovider','--timeout=180','--basetemp='+str(p/(label+'-tmp')),'--junitxml='+str(p/(label+'.xml'))]
start=time.monotonic()
with (p/(label+'.log')).open('w') as f:r=subprocess.run(argv,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(p)+os.pathsep+str(Path.cwd()),'UI18_EVIDENCE':str(p)},stdout=f,stderr=subprocess.STDOUT,timeout=220)
(p/(label+'.json')).write_text(json.dumps({'argv':argv,'cwd':str(Path.cwd()),'code':r.returncode,'elapsed':time.monotonic()-start},indent=2))
print((p/(label+'.log')).read_text()[-4000:]);sys.exit(r.returncode)
