import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(__file__).parent
name, *nodes = sys.argv[1:]
argv=['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','pytest','-q','-p','no:cacheprovider',*nodes,'--basetemp='+str(root/(name+'-tmp'))]
start=time.time()
with (root/(name+'.log')).open('w') as output:
    result=subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
(root/(name+'.json')).write_text(json.dumps({'argv':argv,'exit_code':result.returncode,'seconds':time.time()-start},indent=2)+'\n')
print((root/(name+'.log')).read_text()[-14000:])
sys.exit(result.returncode)
