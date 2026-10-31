from pathlib import Path
import json,subprocess,sys,time
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c')
SOCKET=str(ROOT/'tmux.sock')
TARGET='consoledev:0.0'
argv=sys.argv[1:]
operation=argv[0]
args=argv[1:]
def call(cmd):
 p=subprocess.run(['tmux','-S',SOCKET,*cmd],text=True,capture_output=True)
 assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
 return p.stdout
if operation=='capture':
 label=args[0]
 assert label.replace('-','').isalnum()
 text=call(['capture-pane','-p','-t',TARGET])
 ansi=call(['capture-pane','-p','-e','-t',TARGET])
 (ROOT/'evidence'/f'{label}.txt').write_text(text)
 (ROOT/'evidence'/f'{label}.ansi.txt').write_text(ansi)
 print(text)
elif operation=='text':
 call(['send-keys','-t',TARGET,'-l',args[0]])
elif operation=='keys':
 call(['send-keys','-t',TARGET,*args])
elif operation=='click':
 col,row=map(int,args)
 assert 1<=col<=235 and 1<=row<=52
 call(['send-keys','-t',TARGET,'-l',f'\x1b[<0;{col};{row}M\x1b[<0;{col};{row}m'])
else:
 raise SystemExit('unknown operation')
with (ROOT/'evidence'/'operations.jsonl').open('a') as f:
 f.write(json.dumps({'operation':operation,'args':args,'time':time.time()})+'\n')
