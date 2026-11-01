from pathlib import Path
import subprocess,sys
r=Path(__file__).parent
base=['/opt/homebrew/bin/tmux','-S',str(r/'tmux.sock')]
action=sys.argv[1]
if action=='cap':
 label=sys.argv[2]
 for ansi,suffix in [(False,'.txt'),(True,'.ansi')]:
  data=subprocess.check_output(base+['capture-pane','-p']+(['-e'] if ansi else [])+['-t','followup'])
  (r/(label+suffix)).write_bytes(data)
 print((r/(label+'.txt')).read_text())
elif action in ('click','paste','scroll'):
 if action=='click':
  x,y=map(int,sys.argv[2:4]);data=f'\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m'.encode()
 elif action=='scroll':data=b'\x1b[<64;105;20M'*int(sys.argv[2])
 else:data=b'\x1b[200~'+sys.argv[2].encode()+b'\x1b[201~'
 subprocess.run(base+['send-keys','-t','followup','-H',*[f'{b:02x}' for b in data]],check=True)
elif action=='key':subprocess.run(base+['send-keys','-t','followup',sys.argv[2]],check=True)
else:raise ValueError(action)
