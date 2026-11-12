import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).parent
session=sys.argv[1]
case=root/('rag-only' if session.startswith('rag') else 'direct-ready' if session=='direct-ready' else 'direct')
base=['tmux','-L','archive-profilev2-20261018']
def call(*args):return subprocess.check_output(base+list(args))
action=sys.argv[2]
if action=='snapshot':
    phase=sys.argv[3]
    plain=call('capture-pane','-p','-t',session)
    ansi=call('capture-pane','-p','-e','-t',session)
    for ext,data in [('txt',plain),('ansi',ansi)]:
        p=case/(phase+'.'+ext)
        assert not p.exists(),p
        p.write_bytes(data)
    (case/(phase+'-terminal.json')).write_bytes(call('list-panes','-t',session,'-F','#{pane_id} #{pane_width} #{pane_height} #{pane_dead} #{pane_dead_status}'))
    print('\n'.join(f'{i:02} {line}' for i,line in enumerate(plain.decode().splitlines(),1)))
elif action=='prepare':
    def screen():return call('capture-pane','-p','-t',session).decode()
    def wait(text):
        for _ in range(100):
            current=screen()
            if text in current:return current
            time.sleep(.2)
        raise AssertionError('native screen did not contain '+text)
    def save(phase):
        (case/(phase+'.txt')).write_text(screen())
        (case/(phase+'.ansi')).write_bytes(call('capture-pane','-p','-e','-t',session))
    def click(x,y):call('send-keys','-t',session,'-l',f'\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m')
    wait('No messages yet.');save('initial')
    click(8,13);wait('Change Workspace');save('workspace-picker')
    click(71,24);wait('Conversation | Recovery Check Chat');save('workspace-selected')
    click(114,44);wait('Edit system prompt');save('prompt-control')
    click(86,36);wait('System Prompt: off');save('prompt-cleared')
    click(45,46)
    call('set-buffer','-b','archive-profilev2','Find recently archived chats across all workspaces.')
    call('paste-buffer','-p','-b','archive-profilev2','-t',session)
    wait('Pasted text | 51 characters');call('send-keys','-t',session,'Enter')
    wait('Expand?');call('send-keys','-t',session,'Enter')
    current=wait('Find recently archived chats across all workspaces.');save('search-expanded')
    print('Native workspace/policy selected, generated prompt cleared, exact search expanded; not sent.')
    print('\n'.join(f'{i:02} {line}' for i,line in enumerate(current.splitlines(),1) if i in (7,12,44,46)))
elif action=='click':
    x,y=map(int,sys.argv[3:5]);assert 1<=x<=160 and 1<=y<=48
    call('send-keys','-t',session,'-l',f'\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m')
elif action=='key':call('send-keys','-t',session,*sys.argv[3:])
elif action=='type':
    call('set-buffer','-b','archive-profilev2',sys.argv[3])
    call('paste-buffer','-p','-b','archive-profilev2','-t',session)
else:raise SystemExit('unsupported native input action')
