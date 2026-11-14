from pathlib import Path
import json,shlex,subprocess,sys,time
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c')
PYTHON='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
boot=sys.argv[1]
assert boot.isalnum()
argv=['tmux','-S',str(ROOT/'tmux.sock'),'new-session','-d','-s','consoledev','-x','235','-y','52',shlex.join([PYTHON,str(ROOT/'launch.py'),boot])]
p=subprocess.run(argv,text=True,capture_output=True)
(ROOT/'evidence'/(boot+'-tmux.json')).write_text(json.dumps({'argv':argv,'time':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n')
print(p.stdout+p.stderr,end='')
sys.exit(p.returncode)
