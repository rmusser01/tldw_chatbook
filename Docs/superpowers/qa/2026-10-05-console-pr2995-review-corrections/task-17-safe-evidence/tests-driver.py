import hashlib,json,os,subprocess,tempfile,time,sys
from pathlib import Path
wt=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');sdd=wt/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';out=sdd/'task-17-safe-evidence';selection=json.loads((sdd/'task-17-latest-dev-preflight-selection.json').read_text()); source=json.loads((out/'source-carry.json').read_text());python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
env=os.environ.copy();env['PYTHONPATH']=str(wt)
mode=sys.argv[1]; groups=selection['test_groups']
for group in groups:
 name=group['name'];run_name=f'{name}-{mode}';base=Path(tempfile.mkdtemp(prefix=f'pr2995-task17-{mode}-'));argv=[python,'-m','pytest','-q' if mode=='collection' else '-vv','-p','no:randomly',f'--basetemp={base}/pytest',*group['selectors']]
 if mode=='collection': argv+=['--collect-only']
 else: argv+=[f'--junitxml={out}/{run_name}.xml']
 receipt={'argv':argv,'cwd':str(wt),'environment_overrides':{'PYTHONPATH':str(wt)},'head':source['integrated_head'],'source_sha256':{p:e['sha256'] for p,e in source['paths'].items()},'canonical_private_profile':'Unmodified Tests/conftest.py; existing bootstrap_profile markers; no extra plugin or warning override.','timeout_seconds':300,'expected_nodes':group['selectors'],'expected_count':group['expected_cases_from_source'],'recorded_before_execution':True}
 (out/f'{run_name}-argv.json').write_text(json.dumps(receipt,indent=2)+'\n')
 started=time.monotonic()
 with (out/f'{run_name}.stdout').open('w') as stdout,(out/f'{run_name}.stderr').open('w') as stderr:
  p=subprocess.Popen(argv,cwd=wt,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  try:code=p.wait(timeout=300)
  except subprocess.TimeoutExpired:
   import signal
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=124
 output=(out/f'{run_name}.stdout').read_text(); errors=(out/f'{run_name}.stderr').read_text();(out/f'{run_name}.log').write_text(output+'\n--- STDERR ---\n'+errors)
 result={'argv':argv,'exit':code,'elapsed_seconds':time.monotonic()-started,'stdout':f'{run_name}.stdout','stderr':f'{run_name}.stderr','log':f'{run_name}.log','complete_outputs':True,'pid':p.pid,'closed_process':p.poll() is not None,'head_after':subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip(),'source_after_unchanged':all(hashlib.sha256((wt/path).read_bytes()).hexdigest()==e['sha256'] for path,e in source['paths'].items())}
 if mode=='collection':
  nodes=[line.strip() for line in output.splitlines() if line.startswith('Tests/') and '::' in line];result['collected_nodes']=nodes;result['exact_selection_equal']=nodes==group['selectors'];result['collected_count']=len(nodes)
 else:
  import xml.etree.ElementTree as ET
  xml=out/f'{run_name}.xml';result['junit']=xml.name if xml.exists() else None
  if xml.exists():
   tree=ET.parse(xml);cases=list(tree.iter('testcase'));result['cases']=[dict(x.attrib) for x in cases];result['case_count']=len(cases);result['failed_cases']=[dict(x.attrib) for x in cases if x.find('failure') is not None or x.find('error') is not None];result['skipped_cases']=[dict(x.attrib) for x in cases if x.find('skipped') is not None]
 (out/f'{run_name}-result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(name,mode,'exit',code,'elapsed',round(result['elapsed_seconds'],2),flush=True)
 print('\n'.join(output.splitlines()[-9:]),flush=True)
 assert result['head_after']==source['integrated_head'] and result['source_after_unchanged']
 if code or (mode=='collection' and not result['exact_selection_equal']):
  print('STOP: frozen failure or selection mismatch; no next group, comparison or repair',flush=True);sys.exit(1)
