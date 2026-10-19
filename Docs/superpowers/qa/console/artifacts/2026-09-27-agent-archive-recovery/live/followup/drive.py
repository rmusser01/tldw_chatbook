import pathlib,subprocess,sys,time
R=pathlib.Path(__file__).parent
base=['tmux','-S',str(R/'tmux.sock')]
def run(*args,input=None):return subprocess.run(base+list(args),input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
op=sys.argv[1]
if op=='cap':
 text=run('capture-pane','-p','-t','followup').decode();(R/(sys.argv[2]+'.txt')).write_text(text)
 (R/(sys.argv[2]+'.ansi')).write_bytes(run('capture-pane','-e','-p','-t','followup'));print(text)
elif op=='paste':
 text=sys.argv[2];run('load-buffer','-',input=text.encode());run('paste-buffer','-p','-t','followup')
elif op=='click':
 c=int(sys.argv[2]);r=int(sys.argv[3]);seq=f'\x1b[<0;{c};{r}M\x1b[<0;{c};{r}m';run('send-keys','-H','-t','followup',*[f'{b:02x}' for b in seq.encode()])
elif op=='key':run('send-keys','-t','followup',*sys.argv[2:])
