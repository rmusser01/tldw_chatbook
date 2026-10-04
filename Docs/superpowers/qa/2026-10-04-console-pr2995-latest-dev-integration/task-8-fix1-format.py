import ast,difflib,hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'
sys.path.insert(0,str(root/'scripts/terminal_qualification'))
import format_ratchet as ratchet
ratchet._git=lambda repo,*args:subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=repo)
ratchet._repo_root=lambda:root
paths=['tldw_chatbook/Chat/console_chat_controller.py','Tests/Chat/test_console_chat_create_integration.py']
baseline=out/'task-8-fix1-formatter-baseline.json'
if sys.argv[1]=='snapshot':
 ratchet.snapshot(base='72c67b80a870308e2425b5c2ad9437c3cfe01bd8',output=baseline,paths=paths,replace=False)
elif sys.argv[1]=='verify':ratchet.verify(baseline=baseline,head=None)
else:
 proof=[]
 for path in paths:
  p=root/path;before=p.read_text();old=before.splitlines(keepends=True)
  formatted=subprocess.check_output([sys.executable,'-m','ruff','format','--stdin-filename',path,'-'],input=before.encode()).decode().splitlines(keepends=True)
  diff=subprocess.check_output(['git','-c','gc.auto=0','diff','--unified=0','72c67b80a870308e2425b5c2ad9437c3cfe01bd8','--',path],text=True)
  ranges=[(int(a)-1,int(a)-1+max(1,int(b or '1'))) for a,b in re.findall(r'^@@ .*? \+(\d+)(?:,(\d+))? @@',diff,re.M)]
  edits=[]
  for tag,a,b,c,d in difflib.SequenceMatcher(None,old,formatted,autojunk=False).get_opcodes():
   if tag!='equal' and any(a<=end and b>=start for start,end in ranges):edits.append((a,b,formatted[c:d]))
  for a,b,new in reversed(edits):old[a:b]=new
  after=''.join(old)
  assert ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False)
  p.write_text(after);proof.append(dict(path=path,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),full_module_ast_exact=True,hunks=len(edits)))
 (out/(sys.argv[2] if len(sys.argv)>2 else 'task-8-fix1-format-proof.json')).write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps(proof))
