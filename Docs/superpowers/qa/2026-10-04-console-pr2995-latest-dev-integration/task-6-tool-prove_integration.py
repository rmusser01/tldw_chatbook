import ast,subprocess,pathlib,json,hashlib,difflib
R=pathlib.Path.cwd(); O=R/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'; old='b265d5bd2c2f5a28e4f56c2115f6247988d29ac2'; base='81c7c94f486491d6fde09584710dc3a3172225a0';dev='ca2992cb10b24307fbae050643472ffb0a4388e7'
def g(*a):return subprocess.check_output(['git',*a])
def tree(ref):return {p.decode():sha.decode() for row in g('ls-tree','-r',ref).splitlines() for attrs,p in [row.split(b'\t',1)] for mode,typ,sha in [attrs.split()]}
def changed(a,b):return set(g('diff','--name-only',a,b).decode().splitlines())
def blob(ref,p):return g('show',ref+':'+p)
def asts(data):return ast.dump(ast.parse(data),include_attributes=False)
a,b,c=tree(old),tree(dev),tree('HEAD'); up=changed(base,dev); owned=changed(base,old)
non=[{'path':p,'dev':b.get(p),'head':c.get(p),'equal':b.get(p)==c.get(p)} for p in sorted(up-owned)]
qa=[{'path':p,'old':a[p],'head':c.get(p),'equal':a[p]==c.get(p)} for p in sorted(a) if p.startswith('qa/')]
py=[]
for p in sorted(owned):
 if not p.endswith('.py') or p not in a or p not in c:continue
 x,y=blob(old,p),blob('HEAD',p);eq=asts(x)==asts(y); item={'path':p,'old_blob':a[p],'head_blob':c[p],'old_sha256':hashlib.sha256(x).hexdigest(),'head_sha256':hashlib.sha256(y).hexdigest(),'strict_ast_equal':eq,'upstream_overlap':p in up}
 if not eq:
  name='task-6-canonical-'+p.replace('/','--')+'.diff';(O/name).write_text(''.join(difflib.unified_diff(ast.unparse(ast.parse(x)).splitlines(True),ast.unparse(ast.parse(y)).splitlines(True),fromfile=old+':'+p,tofile='HEAD:'+p)));item['diff']=name
 py.append(item)
oldcomm=g('rev-list','--reverse',base+'..'+old).decode().splitlines();newcomm=g('rev-list','--reverse',dev+'..HEAD').decode().splitlines()
mapping=[{'old':x,'new':y,'subject_match':g('show','-s','--format=%s',x)==g('show','-s','--format=%s',y)} for x,y in zip(oldcomm,newcomm)]
result={'old':old,'base':base,'dev':dev,'head':g('rev-parse','HEAD').decode().strip(),'ancestor_exit':subprocess.run(['git','merge-base','--is-ancestor',dev,'HEAD']).returncode,'nonoverlap_upstream':non,'historical_qa':qa,'owned_python':py,'old_commit_count':len(oldcomm),'new_commit_count':len(newcomm),'mapping':mapping}
(O/'task-6-preservation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'nonoverlap_count':len(non),'nonoverlap_mismatch':[x for x in non if not x['equal']],'qa_count':len(qa),'qa_mismatch':[x for x in qa if not x['equal']],'owned_py':len(py),'semantic_deltas':[x['path'] for x in py if not x['strict_ast_equal']],'commits':len(mapping),'mapping_match':all(x['subject_match'] for x in mapping)},indent=2))
