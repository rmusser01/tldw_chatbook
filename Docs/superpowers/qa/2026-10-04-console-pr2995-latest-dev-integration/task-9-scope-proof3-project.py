"""Assemble a disposable structural projection; never execute it."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
P=Path(__file__).parent
s=(P/'task-9-scope-proof3-controller-union.py').read_text(); lines=s.splitlines(keepends=True)
def nodes(text):
 t=ast.parse(text);c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='ConsoleChatController');return t,c,{n.name:n for n in t.body+c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def start(n):return min([n.lineno]+[d.lineno for d in n.decorator_list])
def fragment(text,n):return ''.join(text.splitlines(keepends=True)[start(n)-1:n.end_lineno])
_,c,ns=nodes(s);p2=json.loads((P/'task-9-scope-proof2-map.json').read_text());sketch=(P/'task-9-scope-proof2-controller.py').read_text();_,pc,pns=nodes(sketch)
comp=(P/'task-9-scope-proof3-compaction-controller.py').read_text();_,cc,cns=nodes(comp)
edits=[]
for item in p2['global_and_documentation_map']:
 name=item['name'];n=ns[name];edits.append((start(n)-1,n.end_lineno,fragment(sketch,pns[name]) if name in pns else ''))
for name,n in cns.items():
 if name=='__init__':continue
 orig=ns[name];edits.append((start(orig)-1,orig.end_lineno,fragment(comp,n)))
old_init=ns['__init__']; replaced=[n for n in old_init.body if 'InterruptRoundHost' in ast.unparse(n) or 'after_remount' in ast.unparse(n)]
assert sum(n.end_lineno-n.lineno+1 for n in replaced)==5
pinit=pns['__init__'];cinit=cns['__init__']
pbody=''.join(sketch.splitlines(keepends=True)[pinit.lineno+1:pinit.end_lineno])
cbody=''.join(comp.splitlines(keepends=True)[cinit.lineno+1:cinit.end_lineno])
assert len(pbody.splitlines())==280 and len(cbody.splitlines())==115
for index,n in enumerate(replaced):edits.append((n.lineno-1,n.end_lineno,pbody if index==0 else ''))
# One real blank line separates the new constructor fragment from existing statements.
edits.append((old_init.end_lineno,old_init.end_lineno,'\n'+cbody))
for a,b,v in sorted(edits,reverse=True):lines[a:b]=[v] if v else []
out=''.join(lines);ast.parse(out);compile(out,'controller structural projection','exec');(P/'task-9-scope-proof3-controller-projected.py').write_text(out)
# A union audit compares unchanged prior-boundary ASTs and records actual incoming overlaps.
base=(P/'task-9-scope-proof3-controller-base.py').read_text();ours=(P/'task-9-scope-proof3-controller-ours.py').read_text();dev=(P/'task-9-scope-proof3-controller-dev.py').read_text()
def inventory(text):
 def descend(body,prefix=''):
  out={}
  for n in body:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    key=prefix+n.name;out[key]=ast.dump(n)
    if isinstance(n,ast.ClassDef):out.update(descend(n.body,key+'.'))
  return out
 return descend(ast.parse(text).body)
b,o,d=map(inventory,[base,ours,dev]);ours_changed={k for k in b.keys()|o.keys() if b.get(k)!=o.get(k)};dev_changed={k for k in b.keys()|d.keys() if b.get(k)!=d.get(k)}
# Source/test routing search is limited to the selected five names and two new adapter names.
names=[n for n in cns if n!='__init__']+['_dispatch_console_trace_recovery','_console_trace_recovery_state']
cmd=['git','grep','-n']
for name in names:cmd+=['-e',name]
cmd+=['f800952214','--','tldw_chatbook','Tests']
r=subprocess.run(cmd,capture_output=True,text=True)
(P/'task-9-scope-proof3-routes.txt').write_text(r.stdout)
measure=json.loads((P/'task-9-scope-proof3-compaction-measure.json').read_text());measure['construction_separator_lines']=1;measure['projected_controller_lines']=len(out.splitlines());measure['headroom']=29367-len(out.splitlines())
(P/'task-9-scope-proof3-compaction-measure.json').write_text(json.dumps(measure,indent=2)+'\n')
sm=json.loads((P/'task-9-scope-proof3-screen-measure.json').read_text());sm['base_methods']=761;(P/'task-9-scope-proof3-screen-measure.json').write_text(json.dumps(sm,indent=2)+'\n')
summary={'controller_projection_lines':len(out.splitlines()),'controller_headroom':29367-len(out.splitlines()),'incoming_changed_definitions':sorted(dev_changed),'ours_changed_definitions':sorted(ours_changed),'overlap_definitions':sorted(ours_changed&dev_changed),'compaction_owner_projected_module_lines':3034+2+measure['owner_class_lines'],'prior_immutable_files_sha256':{str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('task-9-scope-proof*') if not f.name.startswith('task-9-scope-proof3') and f.is_file()}}
(P/'task-9-scope-proof3-projection-map.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['prior_immutable_files_sha256','ours_changed_definitions','incoming_changed_definitions']},indent=2))
