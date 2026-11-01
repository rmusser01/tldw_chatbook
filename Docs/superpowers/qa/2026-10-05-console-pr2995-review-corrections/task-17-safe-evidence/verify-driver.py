import ast,collections,hashlib,json,subprocess,zipfile
from pathlib import Path
wt=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook'); sdd=wt/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'; out=sdd/'task-17-safe-evidence'; s=json.loads((sdd/'task-17-latest-dev-preflight-selection.json').read_text()); base='13d1668d4eedbdfcab50fff28a47a8a68346f033'; selected=s['selected_dev']
def git(*args): return subprocess.check_output(['git',*args],cwd=wt)
def sha(b): return hashlib.sha256(b).hexdigest()
def save(name,data): (out/name).write_text(json.dumps(data,indent=2)+'\n')
def declarations(b):
 t=ast.parse(b); result={}; counts=collections.Counter()
 def visit(body,prefix=''):
  for n in body:
   if isinstance(n,ast.ClassDef): visit(n.body,prefix+n.name+'.')
   elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
    key=prefix+n.name; counts[key]+=1; result[f'{key}@{counts[key]}']=sha(ast.dump(n,include_attributes=False).encode())
 visit(t.body); return result
head=git('rev-parse','HEAD').decode().strip(); sources={}; upstream={}; feature={}
for p in s['incoming_paths']:
 path=p['path']; b=(wt/path).read_bytes(); expected=p['candidate']; actual={'sha256':sha(b),'bytes':len(b),'lines':len(b.splitlines()),'git_blob':git('rev-parse',f'HEAD:{path}').decode().strip()}; assert all(actual[k]==expected[k] for k in expected),(path,actual,expected)
 actual['selected_dev_whole_file_equal']=b==git('show',f'{selected}:{path}'); sources[path]=actual
 if path.endswith('.py'):
  current=declarations(b); native=declarations(git('show',f'{selected}:{path}')); upstream[path]={}
  for key,vals in p.get('incoming_declaration_ast_carry',{}).items():
   assert native.get(key)==vals['selected_dev'],(path,key); assert current.get(key)==vals['candidate'],(path,key); upstream[path][key]={'selected_dev':native.get(key),'actual':current.get(key),'matches_selected_candidate':True,'exact_upstream_AST':current.get(key)==native.get(key)}
  if path in s['three_composed_paths']:
   old=declarations(git('show',f'{base}:{path}')); incoming=p['declarations']['incoming_changed']; unchanged={k:dict(base=v,actual=current.get(k)) for k,v in old.items() if k not in incoming}; assert all(v['base']==v['actual'] for v in unchanged.values()); feature[path]=unchanged
for path,e in s['named_exact_owner_carry'].items():
 b=(wt/path).read_bytes(); assert sha(b)==e['sha256']; sources[path]={'sha256':sha(b),'bytes':len(b),'lines':len(b.splitlines()),'unchanged_named_owner':True}
for path,e in s['caps'].items():
 if path=='screen_methods': continue
 b=(wt/path).read_bytes(); assert sha(b)==e['sha256']; assert len(b.splitlines())==e['actual']; sources[path]={'sha256':sha(b),'bytes':len(b),'lines':len(b.splitlines()),'limit':e['limit']}
for path in ['pyproject.toml','Tests/UI/conftest.py','Tests/UI/app_factory.py','Tests/Architecture/test_screen_size_ratchet.py','Tests/Architecture/test_persistent_diagnostic_inventory.py','tldw_chatbook/Widgets/Console/console_composer_bar.py','tldw_chatbook/app.py','tldw_chatbook/css/tldw_cli_modular.tcss']:
 b=(wt/path).read_bytes(); assert b==git('show',f'{base}:{path}'); sources[path]={'sha256':sha(b),'bytes':len(b),'lines':len(b.splitlines()),'unchanged_profile_budget_or_route_owner':True}
screen=ast.parse((wt/'tldw_chatbook/UI/Screens/chat_screen.py').read_bytes()); cls=next(n for n in screen.body if isinstance(n,ast.ClassDef) and n.name=='ChatScreen'); assert len([n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))])==759
before=git('ls-tree','-r','--full-tree',base).decode().splitlines(); after=git('ls-tree','-r','--full-tree','HEAD').decode().splitlines(); bm={x.split('\t',1)[1]:x.split('\t',1)[0] for x in before}; am={x.split('\t',1)[1]:x.split('\t',1)[0] for x in after}; owned={p['path'] for p in s['incoming_paths']}; delta={p for p in bm.keys()|am.keys() if bm.get(p)!=am.get(p)}; assert delta==owned,(delta-owned,owned-delta)
unowned=sorted((p,bm[p]) for p in bm if p not in owned); unowned_after=sorted((p,am[p]) for p in am if p not in owned); assert unowned==unowned_after
qa={}
for directory,count in s['qa_carry']['current_qa_directory_counts'].items():
 rows=[(p,bm[p]) for p in sorted(bm) if p.startswith(directory)]; now=[(p,am[p]) for p in sorted(am) if p.startswith(directory)]; assert rows==now; assert len(rows)==count; qa[directory]={'rows':count,'before_after_exact_equal':True,'path_mode_blob_sha256':sha(json.dumps(rows,separators=(',',':')).encode()),'preflight_digest_preserved_at_original_algorithm':s['qa_carry']['current_qa_directory_path_blob_digests'][directory]}
z=s['qa_carry']['original_zip']; zip_path=wt/z['path']; b=zip_path.read_bytes(); assert len(b)==z['bytes'] and sha(b)==z['sha256']; entries=len(zipfile.ZipFile(zip_path).infolist()); assert entries==2118
inv='Docs/security/production-diagnostic-inventory.json'; before_inv=json.loads(git('show',f'{base}:{inv}')); after_inv=json.loads((wt/inv).read_bytes()); row=s['diagnostic_union']['selected_dev_row']
def strip_row(value):
 if isinstance(value,list): return [strip_row(x) for x in value if x!=row]
 if isinstance(value,dict): return {k:strip_row(v) for k,v in value.items()}
 return value
assert strip_row(after_inv)==before_inv
commits_before=(out/'feature-commits-before.txt').read_text().splitlines(); commits_after=git('rev-list','--reverse',f'{selected}..HEAD').decode().splitlines(); assert len(commits_before)==len(commits_after)==92
commit_map=[]
for old,new in zip(commits_before,commits_after):
 assert git('show','-s','--format=%s%n%B',old)==git('show','-s','--format=%s%n%B',new); commit_map.append({'before':old,'after':new,'subject':git('show','-s','--format=%s',new).decode().strip()})
assert all(subprocess.run(['git','merge-base','--is-ancestor',c,'HEAD'],cwd=wt).returncode==0 for c in s['incoming_commits'])
save('source-carry.json',{'root_metadata_base':base,'selected_dev':selected,'integrated_head':head,'paths':sources,'incoming_declaration_AST_carry':upstream,'protected_other_composed_declarations':feature,'unchanged_limits':s['unchanged_limits'],'screen_methods':759,'diagnostic_row':row,'diagnostic_remove_only_new_row_matches_base':True})
save('qa-carry.json',{'root_metadata_base':base,'integrated_head':head,'unowned_paths':len(unowned),'unowned_before_after_equal':True,'unowned_path_mode_blob_digest':sha(json.dumps(unowned,separators=(',',':')).encode()),'actual_changed_paths':sorted(delta),'qa_directories':qa,'original_1983_digest':s['qa_carry']['original_digest'],'original_1983_preservation':'Preflight verified exact rows; every QA/unowned base blob remains exact in integrated HEAD. No historical receipt retargeted.','original_zip':dict(z,actual_entries=entries),'public527_carry':'Every historical tracked QA base blob preserved; original public527 and all additive receipts unchanged.'})
save('feature-commit-carry.json',{'all_incoming_10_commits_exact_ancestors':True,'feature_commit_count':92,'mapping':commit_map})
receipt=json.loads((out/'rebase-receipt.json').read_text()); r=next(x for x in receipt if x['argv'][:2]==['git','ls-tree'])
if 'stdout' in r:
 assert r['stdout']==(out/'root-base-path-blobs.txt').read_text(); r['stdout_path']='root-base-path-blobs.txt'; r['stdout_sha256']=sha(r.pop('stdout').encode()); save('rebase-receipt.json',receipt)
print('Verified19 candidate hashes,16 exact-dev files, incoming AST pins, protected declarations/owners,92 feature commits,10 incoming ancestors,all unowned bytes and QA/ZIP; HEAD',head)
