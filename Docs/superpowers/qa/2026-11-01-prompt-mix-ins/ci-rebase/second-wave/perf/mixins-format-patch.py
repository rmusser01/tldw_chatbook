import ast,difflib,subprocess
from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
files=['tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py','tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py','tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py','tldw_chatbook/Widgets/Console/trace_export_dialog.py']
output=['*** Begin Patch']
for name in files:
 old=(root/name).read_text();new=old
 spans=[(n.lineno,n.end_lineno) for n in ast.walk(ast.parse(old)) if isinstance(n,ast.FunctionDef) and n.name=='compose']
 for lo,hi in sorted(spans,reverse=True):
  result=subprocess.run(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff','format','--no-cache','--stdin-filename',str(root/name),f'--range={lo}-{hi+1}','-'],input=new,text=True,capture_output=True,check=True);new=result.stdout
 if old!=new:
  output.append('*** Update File: '+str(root/name));output.extend(list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=3,lineterm=''))[2:])
output.append('*** End Patch');Path('/private/tmp/mixins-format.patch').write_text('\n'.join(output)+'\n')
