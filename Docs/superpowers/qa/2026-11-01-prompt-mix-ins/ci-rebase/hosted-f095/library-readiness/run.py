import pathlib,subprocess,json,os,time,sys
p=pathlib.Path(__file__).parent
label=sys.argv[1]; nodes=sys.argv[2:]
argv=['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','pytest',*nodes,'-q','-p','no:cacheprovider','--timeout=300','--basetemp='+str(p/(label+'-tmp')),'--junitxml='+str(p/(label+'.xml'))]
start=time.monotonic()
with (p/(label+'.log')).open('w') as log:
 r=subprocess.run(argv,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
(p/(label+'.json')).write_text(json.dumps({'argv':argv,'cwd':str(pathlib.Path.cwd()),'exit_code':r.returncode,'elapsed':time.monotonic()-start},indent=2)+'\n')
print((p/(label+'.log')).read_text()[-4000:]);raise SystemExit(r.returncode)
