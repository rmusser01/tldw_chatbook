import json,os,pathlib,subprocess,sys,time
p=pathlib.Path(__file__).parent
name=sys.argv[1];cwd=sys.argv[2];extra=sys.argv[3:]
argv=['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','pytest','Tests/Performance/test_ui_latency_guardrails.py::test_destination_tour_stays_under_switch_budgets','-q','-p','no:cacheprovider','--timeout=300','--basetemp='+str(p/(name+'-tmp')),*extra]
start=time.monotonic()
with (p/(name+'.log')).open('w') as out:
 r=subprocess.run(argv,cwd=cwd,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=out,stderr=subprocess.STDOUT)
(p/(name+'.json')).write_text(json.dumps({'argv':argv,'cwd':cwd,'exit_code':r.returncode,'elapsed':time.monotonic()-start},indent=2)+'\n')
print((p/(name+'.log')).read_text()[-5000:]);sys.exit(r.returncode)
