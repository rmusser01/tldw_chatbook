from pathlib import Path
import json,subprocess,ast,collections
p=Path(__file__).parent
base='963fe12e87ae4d2fb415bff3814943ca448cb0e0'
paths=[s for s in subprocess.check_output(['git','diff',base,'--name-only'],text=True).splitlines() if s.endswith('.py') and not s.startswith('Docs/')]
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
report={}
def duplicates(text):
 tree=ast.parse(text);result={}
 for node in ast.walk(tree):
  if isinstance(node,ast.ClassDef):
   counts=collections.Counter(n.name for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not any(isinstance(d,ast.Attribute) and d.attr in {'setter','deleter'} for d in n.decorator_list))
   result[node.name]={n:c for n,c in counts.items() if c>1}
 return result
for path in paths:
 old=subprocess.check_output(['git','show',base+':'+path],text=True);current=Path(path).read_text();before=duplicates(old);after=duplicates(current);added={c:d for c,d in after.items() if d and d!=before.get(c,{})};assert not added,(path,added)
 def fmt(s):
  r=subprocess.run([ruff,'format','--check','--stdin-filename',path,'-'],input=s,text=True,capture_output=True);return r.returncode
 report[path]={'new_duplicate_methods':added,'formatter_base_exit':fmt(old),'formatter_current_exit':fmt(current)}
(p/'static-structure.json').write_text(json.dumps({'base':base,'files':report},indent=2)+'\n')
print('AST duplicate method delta: zero,',len(paths),'files')
print('Formatter newly failing',[k for k,v in report.items() if v['formatter_current_exit'] and not v['formatter_base_exit']])
