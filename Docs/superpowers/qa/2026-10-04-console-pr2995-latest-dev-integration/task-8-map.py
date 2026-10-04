import ast,hashlib,json,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';base='edaeececfb3733885bb5fec3ea7b5823266dff23';up='5e0341d1ec701865e019eb2fd8a5e2028ab2d474';prior='df2ba424de63576d36d9c3e38387c84285303f1c'
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0',*a])
def tree(rev):
 return {line.split('\t',1)[1]:line.split('\t',1)[0].split()[-1] for line in git('ls-tree','-r',rev).decode().splitlines()}
b,u,d=tree(base),tree(up),tree(prior);head=tree('HEAD')
changed_up={p for p in u.keys()|d.keys() if u.get(p)!=d.get(p)};changed_b={p for p in b.keys()|d.keys() if b.get(p)!=d.get(p)}
paths=sorted(changed_up|changed_b)
def raw(rev,p):
 try:
  if rev and p not in {base:b,up:u,prior:d}.get(rev,{}):return b''
  return git('show',f'{rev}:{p}') if rev else (root/p).read_bytes()
 except (subprocess.CalledProcessError,FileNotFoundError):return b''
def h(s):return hashlib.sha256(s).hexdigest()
def funcs(src):
 if not src:return {}
 t=ast.parse(src);text=src.decode();lines=text.splitlines(keepends=True);result={}
 def visit(node,prefix=''):
  for n in ast.iter_child_nodes(node):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    key=prefix+n.name
    if not isinstance(n,ast.ClassDef):
     start=min([n.lineno]+[x.lineno for x in n.decorator_list]);seg=''.join(lines[start-1:n.end_lineno]);result[key]={'source':h(seg.encode()),'ast':h(ast.dump(n,include_attributes=False).encode()),'line':start}
    visit(n,key+'.')
   else:visit(n,prefix)
 visit(t);return result
shared=[]
for p in sorted((changed_up&changed_b)|{x for x in changed_up if x.endswith('.py')}):
 if not p.endswith('.py'):continue
 maps={label:funcs(raw(rev,p)) for label,rev in [('prior',prior),('feature',base),('upstream',up),('final',None)]}
 rows=[]
 for name in sorted(set().union(*(m.keys() for m in maps.values()))):
  vals={k:v.get(name) for k,v in maps.items()}; changed=[k for k in ['feature','upstream'] if vals[k]!=vals['prior']]
  # line relocation isn't a semantic or source change
  def same(a,b,kind):return (a or {}).get(kind)==(b or {}).get(kind)
  if not all(same(vals['prior'],vals[k],'source') for k in ['feature','upstream','final']):
   rows.append({'name':name,**vals,'feature_ast_changed':not same(vals['prior'],vals['feature'],'ast'),'upstream_ast_changed':not same(vals['prior'],vals['upstream'],'ast'),'final_matches_feature_ast':same(vals['final'],vals['feature'],'ast'),'final_matches_upstream_ast':same(vals['final'],vals['upstream'],'ast')})
 shared.append({'path':p,'methods':rows})
qa={p:v for p,v in b.items() if '/qa/' in p or p.startswith('qa/')};incoming_qa={p:u[p] for p in changed_up if p in u and ('/qa/' in p or p.startswith('qa/'))}
nonoverlap=[{'path':p,'upstream_blob':u.get(p),'final_blob':head.get(p),'equal':u.get(p)==head.get(p)} for p in sorted(changed_up-changed_b)]
# Include every selected source + conftest/decorator dependencies, immutable blob/hash baseline.
pre=json.loads((out/'task-8-preflight.json').read_text());prop=json.loads((out/'task-8-newdev-5e0341-proposal.json').read_text());selected=set(pre['selected_test_sources']+pre['dependency_sources'])|{n.split('::')[0] for group in prop['existing_targeted_node_proposal'].values() for n in group}|{'Tests/real_profile_guard.py','Tests/UI/conftest.py','Tests/Chat/conftest.py','Tests/Chat/test_console_agent_swap.py','Tests/Chat/test_console_skill_script_confirm.py','Tests/UI/app_factory.py'}
pending=list(selected)
while pending:
 q=pending.pop()
 if not q.endswith('.py') or not Path(q).is_file():continue
 try:t=ast.parse(Path(q).read_bytes())
 except SyntaxError:continue
 for node in ast.walk(t):
  modules=[]
  if isinstance(node,ast.Import):modules=[x.name for x in node.names]
  elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]
  for module in modules:
   if module.startswith('Tests.'):
    dep=module.replace('.','/')+'.py'
    if Path(dep).is_file() and dep not in selected:selected.add(dep);pending.append(dep)
 for parent in Path(q).parents:
  dep=str(parent/'conftest.py')
  if dep.startswith('Tests/') and Path(dep).is_file() and dep not in selected:selected.add(dep);pending.append(dep)
source_map=[{'path':p,**{k:{'sha256':h(raw(r,p)), 'blob': {'feature':b,'upstream':u,'prior':d}.get(k,{}).get(p)} for k,r in [('prior',prior),('feature',base),('upstream',up),('final',None)]}} for p in sorted(selected|{p for p in paths if p.startswith('tldw_chatbook/') and p.endswith('.py')})]
carry=json.loads((out/'task-7-final-freeze-map.json').read_text());carry_rows=[{'path':p,'task7_sha256':v,'final_sha256':h(raw(None,p)),'exact':v==h(raw(None,p))} for p,v in carry['owned_sources'].items()]
result=dict(base=base,upstream=up,prior=prior,head=git('rev-parse','HEAD').decode().strip(),nonoverlap=nonoverlap,shared_python=shared,source_and_test_maps=source_map,task7_owned_carry=carry_rows,historical_qa={'count':len(qa),'mismatches':[p for p,v in qa.items() if head.get(p)!=v],'blobs':qa},incoming_qa={'count':len(incoming_qa),'mismatches':[p for p,v in incoming_qa.items() if head.get(p)!=v],'blobs':incoming_qa})
(out/('task-8-'+sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'nonoverlap_mismatches':[x['path'] for x in nonoverlap if not x['equal']],'qa_count':len(qa),'qa_mismatch':result['historical_qa']['mismatches'],'incoming_qa':len(incoming_qa),'task7_changed':[x['path'] for x in carry_rows if not x['exact']]}))
