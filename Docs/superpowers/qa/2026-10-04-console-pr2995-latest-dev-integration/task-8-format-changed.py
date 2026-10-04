import ast,hashlib,json,subprocess,sys,textwrap
from pathlib import Path
out=Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge')
paths=subprocess.check_output(['git','-c','gc.auto=0','diff','--name-only'],text=True).splitlines();proof=[]
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def owners(tree,prefix=''):
 for n in tree.body:
  if isinstance(n,ast.ClassDef):yield from owners(n,prefix+n.name+'.')
  elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):yield prefix+n.name,n
for name in paths:
 if not name.endswith('.py'):continue
 p=Path(name);source=p.read_text();original=subprocess.check_output(['git','-c','gc.auto=0','show','HEAD:'+name],text=True);old=dict(owners(ast.parse(original)));tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[];names=[]
 for key,n in owners(tree):
  if key in old and ast.dump(n,include_attributes=False)==ast.dump(old[key],include_attributes=False):continue
  start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1;end=n.end_lineno;chunk=''.join(lines[start:end]);indent=lines[start][:len(lines[start])-len(lines[start].lstrip())];dedent=textwrap.dedent(chunk)
  r=subprocess.run([sys.executable,'-m','ruff','format','--stdin-filename',name,'-'],input=dedent,text=True,capture_output=True,check=True)
  replacement=textwrap.indent(r.stdout,indent);edits.append((start,end,replacement));names.append(key)
 for start,end,replacement in sorted(edits,reverse=True):lines[start:end]=[replacement]
 final=''.join(lines);assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(final),include_attributes=False),name
 p.write_text(final);proof.append({'path':name,'functions':names,'before_sha256':h(source),'after_sha256':h(final),'full_module_ast_exact':True})
(out/'task-8-format-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
