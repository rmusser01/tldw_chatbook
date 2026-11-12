from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c')
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
PYTHON='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
boot=sys.argv[1]
assert boot.isalnum()
env={k:os.environ[k] for k in ('PATH','LANG','LC_ALL','TERM') if k in os.environ}
env.update({'HOME':str(ROOT/'home'),'USERPROFILE':str(ROOT/'home'),'XDG_CONFIG_HOME':str(ROOT/'home'/'.config'),'XDG_DATA_HOME':str(ROOT/'home'/'.local'/'share'),'XDG_CACHE_HOME':str(ROOT/'home'/'.cache'),'TLDW_CONFIG_PATH':str(ROOT/'config.toml'),'PYTHONPATH':str(WT),'PYTHON_KEYRING_BACKEND':'keyring.backends.null.Keyring','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','NO_PROXY':'127.0.0.1,localhost','TERM':'xterm-256color'})
cmd=[PYTHON,'-m','tldw_chatbook.app','--focus']
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()
receipt={'boot':boot,'sha':sha,'argv':cmd,'cwd':str(ROOT/'cwd'),'env':env,'start_time':time.time(),'launcher_pid':os.getpid()}
(ROOT/'evidence'/f'{boot}-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
p=subprocess.Popen(cmd,cwd=ROOT/'cwd',env=env)
receipt['app_pid']=p.pid
(ROOT/'evidence'/f'{boot}-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
receipt['returncode']=p.wait()
receipt['end_time']=time.time()
(ROOT/'evidence'/f'{boot}-exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(receipt['returncode'])
