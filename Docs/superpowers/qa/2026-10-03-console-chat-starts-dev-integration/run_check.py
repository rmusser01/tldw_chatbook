from pathlib import Path
import json, os, subprocess, sys, time
root=Path(__file__).resolve().parent
label=sys.argv[1]; argv=sys.argv[2:]
env=os.environ.copy(); env['TLDW_TEST_GC_EVERY']='1'
start=time.time()
p=subprocess.run(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
record={'argv':argv,'cwd':str(Path.cwd()),'env':{k:v for k,v in env.items() if k.startswith(('TLDW_','XDG_','PYTHON','HF_','TIKTOKEN_','PYTEST_'))},'returncode':p.returncode,'elapsed_s':time.time()-start,'output':p.stdout}
(root/(label+'.json')).write_text(json.dumps(record,indent=2))
(root/(label+'.log')).write_text(p.stdout)
print(p.stdout[-16000:]); print('Recorded:',label,'returncode:',p.returncode)
sys.exit(p.returncode)
