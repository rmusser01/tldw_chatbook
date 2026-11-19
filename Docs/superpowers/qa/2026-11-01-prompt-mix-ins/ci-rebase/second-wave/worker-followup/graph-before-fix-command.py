import ast,importlib.util,json,pathlib,sys,warnings
repo=pathlib.Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook'); spec=importlib.util.spec_from_file_location('worker_guard',repo/'scripts/check_textual_worker_contract.py'); g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)
out={}
for label,base in [('baseline',pathlib.Path('/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci3346-core-extract-i4xaa7gn/baseline-ba1')),('current',repo)]:
 g.REPO_ROOT=base;g.PACKAGE=base/'tldw_chatbook';collected=[]
 for path in g._source_files():
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');t=ast.parse(path.read_text())
  except (SyntaxError,UnicodeDecodeError):continue
  collected.append(g._collect_module(t,g._rel(path)))
 graph=g._WaitGraph(collected)
 funcs=[f for f in graph.functions if f.name=='_run_agent_reply' and f.module.rel.startswith('tldw_chatbook/Chat/')]
 info=[]
 for f in funcs:
  targets=graph._targets(('self','_select_project_instruction_binding'),f.module,graph._class_of(f),f)
  info.append({'key':f.key,'class_context':graph._class_of(f),'callback_targets':[x.key if isinstance(x,g._Function) else x for x in targets],'callback_sites':sorted(set().union(*(graph._target_sites(x) for x in targets))),'function_sites':sorted(f.sites)})
 roots=[r for r in graph.roots() if ('ConsoleChatStartCoordinator.start->_run =>' in r or 'MobileChatService.send->_run_turn =>' in r) and '_select_project_instruction_binding' in r]
 out[label]={'reply_resolution':info,'two_reported_root_rows':roots}
 print(label,json.dumps(out[label]),flush=True)
pathlib.Path('/private/tmp/mixins-worker-graph-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
