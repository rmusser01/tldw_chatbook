import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path.cwd()
evidence=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-5-evidence'
profile=Path('/private/tmp/pr2995-task5/batch-profile');profile.mkdir(mode=0o700)
paths=['Tests/UI/test_console_runtime_ownership.py','Tests/Chat/test_console_viewless_hooks.py','Tests/Agents/test_install_skill_runtime_tool.py','Tests/Chat/test_console_chat_create_integration.py','tldw_chatbook/app.py','tldw_chatbook/app_navigation.py','tldw_chatbook/UI/Screens/chat_screen.py','tldw_chatbook/Chat/console_runtime.py','tldw_chatbook/UI/Console_Modules/session.py','tldw_chatbook/Chat/console_chat_controller.py','tldw_chatbook/Chat/console_chat_start.py','tldw_chatbook/DB/automatic_work.py']
fingerprints=lambda:{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
cmd=['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','pytest',*paths[:4],'--timeout=300','--tb=short','--basetemp=/private/tmp/pr2995-task5/batch-basetemp','--junitxml='+str(evidence/'batch.xml')]
env=os.environ.copy();env.update(TLDW_TEST_CONFIG_ROOT=str(profile),PYTHONPATH=str(root))
receipt=dict(command=cmd,environment={k:env[k] for k in ('TLDW_TEST_CONFIG_ROOT','PYTHONPATH')},base='c0b251a71422536a3da2be421dd60f79e2cb230a',head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),before=fingerprints(),started=time.time())
with (evidence/'batch.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt.update(exit=r.returncode,elapsed=time.time()-receipt['started'],after=fingerprints())
(evidence/'batch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('exit','elapsed','before','after')},indent=2),flush=True)
raise SystemExit(r.returncode)
