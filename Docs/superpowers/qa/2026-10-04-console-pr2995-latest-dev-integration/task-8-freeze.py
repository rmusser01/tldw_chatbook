import ast,collections,hashlib,json,subprocess,sys
from pathlib import Path
out=Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge')
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],text=True)
def sha(b):return hashlib.sha256(b).hexdigest()
head=git('rev-parse','HEAD').strip();base='edaeececfb3733885bb5fec3ea7b5823266dff23';up='5e0341d1ec701865e019eb2fd8a5e2028ab2d474';prior='df2ba424de63576d36d9c3e38387c84285303f1c';union='bc184be782'
old=git('log','--reverse','--format=%H %s',f'{prior}..{base}').splitlines();new=git('log','--reverse','--format=%H %s',f'{up}..{union}').splitlines();assert len(old)==len(new)==42
mapping=[{'old':a.split()[0],'new':b.split()[0],'subject':a.split(' ',1)[1],'subject_exact':a.split(' ',1)[1]==b.split(' ',1)[1]} for a,b in zip(old,new)];assert all(x['subject_exact'] for x in mapping)
parents=git('rev-list','--parents',f'{up}..HEAD').splitlines()
task7=json.loads((out/'task-7-final-freeze-map.json').read_text());carry=[]
for path,changes in task7['changed_functions'].items():
 if not path.endswith('.py'):continue
 t=ast.parse(Path(path).read_bytes());current={sha(ast.dump(n,include_attributes=False).encode()) for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 for name,entry in changes.items():
  after=entry.get('after')
  if after:carry.append({'path':path,'task7_name':name,'ast_sha256':after['ast_sha256'],'present_exact_in_final':after['ast_sha256'] in current})
assert all(x['present_exact_in_final'] for x in carry),[x for x in carry if not x['present_exact_in_final']]
tree={l.split('\t',1)[1]:l.split('\t',1)[0].split()[-1] for l in git('ls-tree','-r','HEAD').splitlines()};required=json.loads((out/'task-7-final-preservation.json').read_text())['historical_qa'];required_rows=[{'path':r['path'],'required_blob':r['final_blob'],'final_blob':tree.get(r['path'])} for r in required];assert len(required_rows)==11572 and all(x['required_blob']==x['final_blob'] for x in required_rows)
census=[]
for path in ['scripts/textual_await_dom_census.tsv','scripts/textual_wait_push_census.tsv','scripts/ui_pr_gate_census.txt','Tests/Chat/console_fork_owner_census.json']:
 if not Path(path).exists():continue
 def read(rev):
  try:return git('show',rev+':'+path)
  except subprocess.CalledProcessError:return ''
 texts={'prior':read(prior),'feature':read(base),'upstream':read(up),'final':Path(path).read_text()};sets={k:set(x for x in v.splitlines() if x.strip() and not x.startswith('#')) for k,v in texts.items()};expected=(sets['feature']|sets['upstream'])-((sets['prior']-sets['feature'])|(sets['prior']-sets['upstream']));census.append({'path':path,'source_sha256':{k:sha(v.encode()) for k,v in texts.items()},'exact_owner_union':sets['final']==expected,'missing':sorted(expected-sets['final']),'extra':sorted(sets['final']-expected),'upstream_moved_removed':sorted(sets['prior']-sets['upstream']),'upstream_moved_added':sorted(sets['upstream']-sets['prior'])})
owned=git('diff','--name-only',union,'HEAD').splitlines();allmaps=[]
for p in owned:
 oldsrc=git('show',union+':'+p);final=Path(p).read_text();t=ast.parse(final);o=ast.parse(oldsrc)
 def functions(tree):
  return {n.name: {'ast':sha(ast.dump(n,include_attributes=False).encode()),'line':n.lineno,'assertions':collections.Counter(ast.dump(a,include_attributes=False) for a in ast.walk(n) if isinstance(a,ast.Assert))} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 before,after=functions(o),functions(t);rows=[]
 for name in before.keys()|after.keys():
  if before.get(name,{}).get('ast')==after.get(name,{}).get('ast'):continue
  b=before.get(name,{});a=after.get(name,{});rows.append({'name':name,'before_ast':b.get('ast'),'after_ast':a.get('ast'),'line':a.get('line'),'removed_assertions':list((b.get('assertions',collections.Counter())-a.get('assertions',collections.Counter())).elements()),'added_assertions':list((a.get('assertions',collections.Counter())-b.get('assertions',collections.Counter())).elements())})
 allmaps.append({'path':p,'union_sha256':sha(oldsrc.encode()),'final_sha256':sha(final.encode()),'changed_functions_and_assertions':rows})
result={'head':head,'base':base,'upstream':up,'rebase_union':git('rev-parse',union).strip(),'upstream_is_ancestor':subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',up,head]).returncode==0,'commit_mapping':mapping,'parents':parents,'source_correction_commit':head,'task7_changed_function_carry':carry,'required_historical_qa':required_rows,'census':census,'corrective_source_and_fixture_map':allmaps,'status':git('status','--porcelain=v1'),'final_tree':tree}
(out/'task-8-final-freeze-map.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'head':head,'commit_pairs':len(mapping),'task7_method_carry':len(carry),'qa':len(required_rows),'census_union':[(x['path'],x['exact_owner_union']) for x in census],'status':result['status']}))
