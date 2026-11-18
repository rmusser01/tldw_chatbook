import os,subprocess,pathlib,time,json,sys
p=pathlib.Path(__file__).parent
argv=["/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python","-m","pytest",*sys.argv[2:],"-vv","-p","no:cacheprovider","-p","probe","--timeout=180","--basetemp="+str(p/(sys.argv[1]+"-tmp")),"--junitxml="+str(p/(sys.argv[1]+".xml"))]
env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(p)+os.pathsep+str(pathlib.Path.cwd()),"UI28_EVIDENCE":str(p)}
start=time.monotonic()
with (p/(sys.argv[1]+".log")).open("w") as f:
 child=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT)
 try:code=child.wait(timeout=230)
 except subprocess.TimeoutExpired:
  child.kill();code=child.wait()
(p/(sys.argv[1]+".json")).write_text(json.dumps({"argv":argv,"cwd":str(pathlib.Path.cwd()),"code":code,"elapsed":time.monotonic()-start},indent=2))
print((p/(sys.argv[1]+".log")).read_text()[-6000:]);sys.exit(code)
