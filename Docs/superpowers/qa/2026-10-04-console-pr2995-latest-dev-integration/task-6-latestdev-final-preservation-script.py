import ast,subprocess,json,hashlib
from pathlib import Path
s=Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge');base='dc4df3dc80f767de7c94a833b0402dd880af235c';olddev='ca2992cb10b24307fbae050643472ffb0a4388e7';dev='67fc5310471823262eb6ce344a539e699784bed8'
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0',*a])
def tree(ref):return {p.decode():a.split()[2].decode() for row in git('ls-tree','-r',ref).splitlines() for a,p in [row.split(b'\t',1)]}
def changed(a,b):return set(git('diff','--name-only',a,b).decode().splitlines())
def blob(ref,p):return git('show',ref+':'+p)
def canon(b):return ast.dump(ast.parse(b),include_attributes=False)
a,b,c=tree(base),tree(dev),tree('HEAD');owned=changed(olddev,base);up=changed(olddev,dev)
head=git('rev-parse','HEAD').decode().strip();proof={'base':base,'olddev':olddev,'dev':dev,'head':head,'ancestor_exit':subprocess.run(['git','merge-base','--is-ancestor',dev,'HEAD']).returncode,'conflicts':['backlog/decisions/README.md','backlog/docs/lessons-backlog-hygiene.md']}
proof['upstream_nonoverlap']=[{'path':p,'dev':b.get(p),'head':c.get(p),'exact':b.get(p)==c.get(p)} for p in sorted(up-owned)];assert all(x['exact'] for x in proof['upstream_nonoverlap'])
proof['owned_python']=[{'path':p,'base':a[p],'head':c.get(p),'byte_equal':a[p]==c.get(p),'strict_ast_equal':canon(blob(base,p))==canon(blob('HEAD',p))} for p in sorted(owned) if p.endswith('.py') and p in a and p in c];assert all(x['byte_equal'] and x['strict_ast_equal'] for x in proof['owned_python'] if x['path'] != 'Tests/UI/test_console_store_continuity.py')
proof['owned_source_all_types']=[{'path':p,'base':a.get(p),'head':c.get(p),'exact':a.get(p)==c.get(p)} for p in sorted(owned) if p.startswith(('tldw_chatbook/','Tests/','scripts/'))];assert all(x['exact'] for x in proof['owned_source_all_types'] if x['path'] != 'Tests/UI/test_console_store_continuity.py')
proof['historical_qa']=[{'path':p,'base':a[p],'head':c.get(p),'exact':a[p]==c.get(p)} for p in sorted(a) if p.startswith(('qa/','Docs/superpowers/qa/'))];assert all(x['exact'] for x in proof['historical_qa'])
proof['owned_non_source_delta']=[p for p in sorted(owned) if a.get(p)!=c.get(p)];assert set(proof['owned_non_source_delta'])==set(proof['conflicts']) | {'Tests/UI/test_console_store_continuity.py'}
oldcomm=git('rev-list','--reverse',olddev+'..'+base).decode().splitlines();newcomm=git('rev-list','--reverse',dev+'..HEAD').decode().splitlines();assert len(oldcomm)==25 and len(newcomm)==26; proof['post_rebase_fixture_commit']=newcomm[-1]; newcomm=newcomm[:25]
proof['mapping']=[{'old':x,'new':y,'subject_match':git('show','-s','--format=%s',x)==git('show','-s','--format=%s',y)} for x,y in zip(oldcomm,newcomm)];assert all(x['subject_match'] for x in proof['mapping'])
proof['test_only_exception']={'path':'Tests/UI/test_console_store_continuity.py','proof':'task-6-latestdev-navigation-fixture-final-proof.json','scope':'existing private profile helper and early manual mount ownership; original navigation assertions preserved'}
proof['all_postcheckpoint_deltas']=sorted(changed(base,'HEAD'))
(s/'task-6-latestdev-final-preservation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:len(proof[k]) for k in ['upstream_nonoverlap','owned_python','owned_source_all_types','historical_qa','mapping']},indent=2))
