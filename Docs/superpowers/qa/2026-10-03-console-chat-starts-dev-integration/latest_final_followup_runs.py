from pathlib import Path
import json,subprocess,sys
root=Path(__file__).resolve().parent
argv=json.loads((root/'baseline21-final.json').read_text())['argv']
assert len([x for x in argv if '::' in x])==21
argv=['--basetemp=/private/tmp/console-latest-final-baseline21-01a0fa6c' if x.startswith('--basetemp=') else x for x in argv]
commands=[('latest-final-baseline21',argv),('latest-final-bootstrap-create-owners',[sys.executable,'-B','-m','pytest','-q','--basetemp=/private/tmp/console-latest-final-bootstrap-create-01a0fa6c','Tests/test_real_profile_guard.py','Tests/Chat/test_console_chat_create_confirm.py','Tests/Chat/test_console_chat_create_integration.py'])]
results=[]
for label,argv in commands:
 results.append(subprocess.run([sys.executable,'-B',str(root/'run_check.py'),label,*argv]).returncode)
raise SystemExit(max(results))
