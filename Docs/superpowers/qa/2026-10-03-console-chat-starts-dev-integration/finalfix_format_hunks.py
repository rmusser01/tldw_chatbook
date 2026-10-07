from pathlib import Path
import ast, hashlib, json, os, re, subprocess, sys
root=Path(__file__).resolve().parent
paths=['Tests/Chat/test_console_chat_create_integration.py', 'Tests/UI/test_console_runtime_ownership.py', 'tldw_chatbook/Chat/console_agent_bridge.py', 'tldw_chatbook/Chat/console_chat_controller.py', 'tldw_chatbook/Chat/console_chat_store.py', 'tldw_chatbook/UI/Console_Modules/session.py']
records=[]
for path in paths:
 before=Path(path).read_bytes()
 diff=subprocess.check_output(['git','diff','--unified=0','--',path],text=True)
 ranges=[]
 for line in diff.splitlines():
  match=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',line)
  if match:
   start=int(match[1]);size=int(match[2] or 1)
   if size:ranges.append((start,start+size-1))
 for start,end in reversed(ranges):
  argv=[sys.executable,'-B','-m','ruff','format','--range',f'{start}-{end}',path]
  result=subprocess.run(argv,capture_output=True,text=True)
  records.append({'argv':argv,'returncode':result.returncode,'output':result.stdout+result.stderr})
  assert result.returncode==0,records[-1]
 after=Path(path).read_bytes()
 assert ast.dump(ast.parse(before))==ast.dump(ast.parse(after)),path
 records.append({'path':path,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'ast_equal':True})
(root/'finalfix-format-hunks-detail.json').write_text(json.dumps(records,indent=2))
print(f'Formatted added/changed ranges in {len(paths)} owned paths; AST equality verified.')
