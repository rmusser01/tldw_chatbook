import ast,subprocess,json,hashlib
from pathlib import Path
s=Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge');base='fb43008890e6e406505ede734633b01402bab4ec';olddev='67fc5310471823262eb6ce344a539e699784bed8';dev='f1f80847a410525ce1b00d27ea0e8be824a98422';fixture='Tests/UI/test_console_store_continuity.py';inventory='Docs/security/production-diagnostic-inventory.json'
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0',*a])
def tree(ref):return {p.decode():a.split()[2].decode() for row in git('ls-tree','-r',ref).splitlines() for a,p in [row.split(b'\t',1)]}
def changed(a,b):return set(git('diff','--name-only',a,b).decode().splitlines())
def blob(ref,p):return git('show',ref+':'+p)
def canon(b):return ast.dump(ast.parse(b),include_attributes=False)
a,b,c=tree(base),tree(dev),tree('HEAD');owned=changed(olddev,base);up=changed(olddev,dev);head=git('rev-parse','HEAD').decode().strip()
proof={'base':base,'olddev':olddev,'dev':dev,'head':head,'ancestor_exit':subprocess.run(['git','merge-base','--is-ancestor',dev,'HEAD']).returncode,'conflicts':[],'overlaps':sorted(owned&up)}
repair_paths={'Tests/UI/test_library_conversation_reader.py','Tests/UI/test_library_conversation_reader_freshness.py','tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py','tldw_chatbook/UI/Library_Modules/library_conversations_state.py','tldw_chatbook/UI/Library_Modules/library_skills_controller.py'}
proof['upstream_nonoverlap']=[{'path':p,'dev':b.get(p),'head':c.get(p),'exact':b.get(p)==c.get(p)} for p in sorted(up-owned)];assert all(x['exact'] for x in proof['upstream_nonoverlap'] if x['path'] not in repair_paths)
proof['owned_python']=[{'path':p,'base':a[p],'head':c.get(p),'byte_equal':a[p]==c.get(p),'strict_ast_equal':canon(blob(base,p))==canon(blob('HEAD',p))} for p in sorted(owned) if p.endswith('.py') and p in a and p in c];assert all(x['byte_equal'] and x['strict_ast_equal'] for x in proof['owned_python'] if x['path']!=fixture)
proof['owned_source_all_types']=[{'path':p,'base':a.get(p),'head':c.get(p),'exact':a.get(p)==c.get(p)} for p in sorted(owned) if p.startswith(('tldw_chatbook/','Tests/','scripts/'))];assert all(x['exact'] for x in proof['owned_source_all_types'] if x['path']!=fixture)
proof['historical_qa']=[{'path':p,'base':a[p],'head':c.get(p),'exact':a[p]==c.get(p)} for p in sorted(a) if p.startswith(('qa/','Docs/superpowers/qa/'))];assert all(x['exact'] for x in proof['historical_qa'])
oldcomm=git('rev-list','--reverse',olddev+'..'+base).decode().splitlines();newcomm=git('rev-list','--reverse',dev+'..HEAD').decode().splitlines();assert len(oldcomm)==27 and len(newcomm)==28
proof['mapping']=[{'old':x,'new':y,'subject_match':git('show','-s','--format=%s',x)==git('show','-s','--format=%s',y)} for x,y in zip(oldcomm,newcomm[:27])];assert all(x['subject_match'] for x in proof['mapping']);proof['post_rebase_fixture_commit']=newcomm[-1]
original=json.loads(blob(base,inventory));incoming=json.loads(blob(dev,inventory));current=json.loads(blob('HEAD',inventory));newpath='tldw_chatbook/UI/Library_Modules/library_conversation_reader_freshness.py';entry=next(x for x in incoming['owners'] if x['path']==newpath);expected=json.loads(json.dumps(original));expected['owners'].append(entry);expected['owners'].sort(key=lambda x:x['path']);assert expected==current
proof['diagnostic_union']={'only_new_owner':entry,'all_other_entries_metadata_exact_base':True,'entry_exact_dev':True,'regeneration_needed':False}
proof['test_only_exception']={'path':fixture,'proof':'task-6-reader-navigation-assertion-proof.json','scope':'two explicit active/visible first-session assertions; original body and earlier guards intact'}
proof['all_postcheckpoint_deltas']=sorted(changed(base,'HEAD'));assert set(proof['all_postcheckpoint_deltas'])==up|{fixture,'tldw_chatbook/UI/Library_Modules/library_skills_controller.py'}
proof['explicit_repair_exceptions']=[{'path':p,'dev':b.get(p),'head':c.get(p),'strict_AST_equal_dev':canon(blob(dev,p))==canon(blob('HEAD',p)),'proof':'task-6-reader-repair-proof.json'} for p in sorted(repair_paths)]
assert all(x['strict_AST_equal_dev'] for x in proof['explicit_repair_exceptions'] if x['path']!='Tests/UI/test_library_conversation_reader.py')
(s/'task-6-reader-final-preservation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:len(proof[k]) for k in ['upstream_nonoverlap','owned_python','owned_source_all_types','historical_qa','mapping']},indent=2))
