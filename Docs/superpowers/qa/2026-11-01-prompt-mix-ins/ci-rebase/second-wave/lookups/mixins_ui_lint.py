import ast, subprocess, json
from pathlib import Path
from collections import Counter
root=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
source=Path('/private/tmp/mixins_ui_assignments.py').read_text();env={};exec(source[source.index('sites='):source.index('for rel,wanted')],env)
files=['tldw_chatbook/'+p for p in env['sites']]+['tldw_chatbook/UI/Console_Modules/composer_run_controls.py','tldw_chatbook/UI/Console_Modules/emergency_stop_hold.py','tldw_chatbook/UI/Console_Modules/prompts.py','tldw_chatbook/UI/Screens/chat_screen.py','Tests/UI/test_plugin_operations.py','Tests/UI/test_console_emergency_stop_ownership.py','Tests/UI/Library_Modules/test_library_notes_sync_attention_listener.py']
Path('/private/tmp/mixins-ui-lookup-owned-files.txt').write_text('\n'.join(files)+'\nTests/Architecture/app_lookup_census.tsv\n')
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';new=[];totals=Counter()
for f in files:
 cur=(root/f).read_text();old=subprocess.check_output(['git','show','HEAD:'+f],cwd=root,text=True)
 def check(s):
  r=subprocess.run([ruff,'check','--output-format=json','--stdin-filename',f,'-'],cwd=root,input=s,text=True,capture_output=True)
  return json.loads(r.stdout)
 before=check(old);after=check(cur);b=Counter((x['code'],x['message']) for x in before);a=Counter((x['code'],x['message']) for x in after)
 totals.update(before=len(before),after=len(after));delta=a-b
 for x in after:
  k=x['code'],x['message']
  if delta[k]:new.append((f,x['location'],*k));delta[k]-=1
print(dict(totals));print('New diagnostics',len(new));print(json.dumps(new,indent=2))
