import ast,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
root=Path.cwd();sdd=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'
BASE='72c67b80a870308e2425b5c2ad9437c3cfe01bd8'; OLD='40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args])
def sha(b):return hashlib.sha256(b).hexdigest()
def old(rev,path):
 r=subprocess.run(['git','-c','gc.auto=0','show',f'{rev}:{path}'],capture_output=True)
 return r.stdout if r.returncode==0 else None
def facts(b):return None if b is None else dict(sha256=sha(b),bytes=len(b),lines=len(b.splitlines()))
def imports(b):
 tree=ast.parse(b);return sha(json.dumps([ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]).encode())
def funcs(b):
 if b is None:return {}
 text=b.decode();lines=text.splitlines(keepends=True);out={}
 def visit(node,prefix=''):
  for n in node.body:
   if isinstance(n,ast.ClassDef):visit(n,prefix+n.name+'.')
   elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
    start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
    out[prefix+n.name]=dict(ast_sha256=sha(ast.dump(n,include_attributes=False).encode()),source_sha256=sha(''.join(lines[start:n.end_lineno]).encode()),line=n.lineno)
 visit(ast.parse(text));return out
def tree(rev):
 return {line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in git('ls-tree','-r',rev).decode().splitlines()}
head=git('rev-parse','HEAD').decode().strip();status=git('status','--porcelain').decode();assert not status
changed=git('diff','--name-only',BASE,head).decode().splitlines();assert changed==['Tests/Chat/test_console_chat_create_integration.py','tldw_chatbook/Chat/console_chat_controller.py']
previous=json.loads((sdd/'task-8-final-preservation.json').read_text());priorfreeze=json.loads((sdd/'task-8-final-freeze-map.json').read_text())
paths={r['path'] for r in previous['source_and_test_maps']}
receipts=[];nodes=set()
for p in sorted(sdd.glob('task-8-fix1-*.json')):
 d=json.loads(p.read_text())
 if isinstance(d,dict) and 'argv' in d and 'exit' in d:
  record=dict(path=str(p),sha256=sha(p.read_bytes()),exit=d['exit'],seconds=d.get('seconds'),argv=d['argv'],before_after_exact=d.get('source_before')==d.get('source_after') if 'source_before' in d else None)
  xml=p.with_suffix('.xml')
  if xml.exists():
   cases=list(ET.parse(xml).getroot().iter('testcase'));record['test_results']=dict(total=len(cases),failed=sum(c.find('failure') is not None for c in cases),errors=sum(c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases))
  receipts.append(record)
  for v in d['argv']:
   if v.startswith('Tests/') or (v.startswith('.superpowers/') and '::' in v):nodes.add(v);paths.add(v.split('::')[0])
  paths.update(d.get('source_before',{}))
paths.update(['tldw_chatbook/Agents/run_context.py','tldw_chatbook/DB/base_db.py','tldw_chatbook/Chat/console_chat_start.py'])
source_maps=[]
for path in sorted(paths):
 p=root/path
 if not p.is_file():raise RuntimeError(path)
 current=p.read_bytes();before=old(OLD,path);checkpoint=old(BASE,path)
 source_maps.append(dict(path=path,original_40b9=facts(before),checkpoint_72c=facts(checkpoint),final=facts(current),unchanged_from_40b9=before==current,unchanged_from_checkpoint=checkpoint==current))
method_maps=[]
for path in changed+['Tests/Chat/test_console_chat_create_confirm.py','Tests/Chat/test_console_chat_start.py','tldw_chatbook/Chat/console_chat_start.py']:
 before=funcs(old(OLD,path));after=funcs((root/path).read_bytes())
 rows=[dict(owner=k,before=before.get(k),after=after.get(k),ast_exact=before.get(k,{}).get('ast_sha256')==after.get(k,{}).get('ast_sha256')) for k in sorted(before.keys()|after.keys())]
 method_maps.append(dict(path=path,functions=rows))
oldtree=tree(OLD);newtree=tree(head);qa={p:b for p,b in oldtree.items() if '/qa/' in p or p.startswith('qa/')};mismatches=[p for p,b in qa.items() if newtree.get(p)!=b];assert not mismatches
required=priorfreeze['required_historical_qa'];print('required row shape',str(required[0])[:200])
source='tldw_chatbook/Chat/console_chat_controller.py';before=old(OLD,source);after=(root/source).read_bytes()
startup=json.loads((sdd/'task-8-final-startup-navigation.json').read_text())
carry=[]
for path,h in startup['source_after'].items():
 now=sha((root/path).read_bytes());carry.append(dict(path=path,qualified_sha256=h,current_sha256=now,exact=h==now,reason='I1 validation methods changed; all imports unchanged, no new module, no startup caller or startup/cap/CSS source change. Covered behavior requalified separately.' if path==source else 'Exact source bytes.'))
# Existing methods changed only in the four scoped validation/confirmation owners.
source_methods=next(r for r in method_maps if r['path']==source)['functions']
changed_methods=[r['owner'] for r in source_methods if r['before'] and not r['ast_exact']]
assert changed_methods==sorted(['ConsoleChatController._chat_creation_record','ConsoleChatController._chat_creation_record_locked','ConsoleChatController._chat_creation_source_live','ConsoleChatController.request_chat_create_confirm'])
assert imports(before)==imports(after)
newtest_methods=next(r for r in method_maps if r['path'].startswith('Tests/Chat/test_console_chat_create_integration'))['functions']
assert all(r['ast_exact'] for r in newtest_methods if r['before'])
observers=['tldw_chatbook/Chat/console_agent_bridge.py','tldw_chatbook/DB/AgentRuns_DB.py','tldw_chatbook/Agents/run_context.py','tldw_chatbook/Chat/console_chat_start.py','tldw_chatbook/Chat/console_chat_store.py','tldw_chatbook/Chat/console_chat_models.py']
for path in observers:assert old(OLD,path)==(root/path).read_bytes()
mapdata=dict(status='DONE_WITH_CONCERNS',head=head,checkpoint=BASE,original_source=OLD,parent=git('rev-parse',head+'^').decode().strip(),tracked_status=status,changed_paths=changed,source_and_all_selected_test_helper_maps=source_maps,selected_nodes=sorted(nodes),complete_function_maps=method_maps,changed_existing_controller_methods=changed_methods,unchanged_runtime_storage_callees=observers,production_imports=dict(before_sha256=imports(before),after_sha256=imports(after),exact=True,new_modules=0),startup_navigation_carry=dict(receipt=str(sdd/'task-8-final-startup-navigation.json'),qualified_cases=58,replayed=False,source_maps=carry,limits=['58 passed with three warnings is carried evidence, not a newly executed startup/nav run.','Preimport source explicitly prewarms chat/controller and excludes that closure from its added-LOC count. Controller imports and all other production source except validation bodies are unchanged.','The newly inspected controller size row is a separate failing gate; none of the 58 previous cases qualified that row.']),cap=dict(path=source,budget=29367,immutable_lines=len(before.splitlines()),current_lines=len(after.splitlines()),inherited_overage=len(before.splitlines())-29367,fix_delta=len(after.splitlines())-len(before.splitlines()),remaining_overage=len(after.splitlines())-29367,immutable_receipt=str(sdd/'task-8-fix1-final-immutable.json'),current_receipt=str(sdd/'task-8-fix1-final-cap.json'),policy='Unchanged; no waiver. Root directed separate structural repair.'),qa_carry=dict(count=len(qa),required_historical_count=len(required),original_tree_sha256=sha(json.dumps(qa,sort_keys=True).encode()),current_same_paths_sha256=sha(json.dumps({p:newtree[p] for p in qa},sort_keys=True).encode()),mismatches=mismatches,prior_complete_map=str(sdd/'task-8-final-freeze-map.json')),prior_evidence=dict(report_sha256=sha((sdd/'task-8-report.md').read_bytes()),review_sha256=sha((sdd/'task-8-review.md').read_bytes()),preservation_sha256=sha((sdd/'task-8-final-preservation.json').read_bytes()),freeze_sha256=sha((sdd/'task-8-final-freeze-map.json').read_bytes())),receipts=receipts,limitations=['Early immutable diagnostic receipts did not themselves hash the imported current integration test file. Final immutable receipt repeats those controls on final selected test source and hashes all 155 inherited selected source/test/helper paths plus its own test helpers before and after. Early receipts remain diagnostic evidence, not sole provenance.','Diagnostic and worker reconstruction ran before the last missing-key correction; that correction only changes nonlogging dict copies and has no import/await/worker/diagnostic sink change. Final derived verification is recorded separately if present.'])
required_blobs={r['path']:r['required_blob'] for r in required}
for label,blobs in [('required_historical',required_blobs),('checkpoint_historical',previous['historical_qa']['blobs']),('incoming',previous['incoming_qa']['blobs'])]:
 mismatches=[p for p,b in blobs.items() if newtree.get(p)!=b]
 assert not mismatches
 mapdata['qa_carry'][label]=dict(count=len(blobs),reference_sha256=sha(json.dumps(blobs,sort_keys=True).encode()),current_sha256=sha(json.dumps({p:newtree[p] for p in blobs},sort_keys=True).encode()),mismatches=mismatches)
task7=[]
for row in priorfreeze['task7_changed_function_carry']:
 hashes={sha(ast.dump(n,include_attributes=False).encode()) for n in ast.walk(ast.parse((root/row['path']).read_bytes())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 exact=row['ast_sha256'] in hashes
 assert exact, row
 task7.append(dict(row,present_exact_in_fix1=exact))
mapdata['task7_34_function_carry']=task7
mapdata['derived_source_carry']=[dict(path=p,original_blob=b,final_blob=newtree.get(p),exact=newtree.get(p)==b) for p,b in oldtree.items() if (p.startswith('tldw_chatbook/css/') or p in [r['path'] for r in priorfreeze['census']] or ('diagnostic' in p and p.startswith('scripts/'))) and newtree.get(p)==b]
(sdd/'task-8-fix1-freeze-map.json').write_text(json.dumps(mapdata,indent=2)+'\n')
(sdd/'task-8-fix1-diff.log').write_bytes(git('diff','--no-ext-diff',BASE,head,'--',*changed))
print(json.dumps({'head':head,'sources':len(source_maps),'cap':mapdata['cap'],'methods_changed':changed_methods,'qa':len(qa)}))
