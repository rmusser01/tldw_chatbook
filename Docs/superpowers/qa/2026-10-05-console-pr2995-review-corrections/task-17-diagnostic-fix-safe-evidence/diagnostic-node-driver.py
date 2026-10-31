import hashlib,json,os,signal,subprocess,tempfile,time
from pathlib import Path
wt=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');sdd=wt/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';out=sdd/'task-17-diagnostic-fix-safe-evidence';source=json.loads((sdd/'task-17-safe-evidence/source-carry.json').read_text());carry=json.loads((out/'inventory-only-carry.json').read_text());base='408453025fc58383b81cb018b01134579cb3141a';python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python';node='Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged'
sources={p:hashlib.sha256((wt/p).read_bytes()).hexdigest() for p in source['paths']};sources['scripts/check_persistent_diagnostic_inventory.py']=hashlib.sha256((wt/'scripts/check_persistent_diagnostic_inventory.py').read_bytes()).hexdigest()
for p,v in source['paths'].items():assert sources[p]==(carry['after_sha256'] if p==carry['path'] else v['sha256'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()==base
assert subprocess.check_output(['git','diff','--name-only'],cwd=wt,text=True).splitlines()==[carry['path']]
env=os.environ.copy();env['PYTHONPATH']=str(wt);temp=Path(tempfile.mkdtemp(prefix='pr2995-task17-diagnostic-fix-'));argv=[python,'-m','pytest','-vv','-p','no:randomly',f'--basetemp={temp}/pytest',node,f'--junitxml={out}/diagnostic-node.xml'];receipt={'argv':argv,'cwd':str(wt),'environment_overrides':{'PYTHONPATH':str(wt)},'head_before_commit':base,'actual_source_sha256':sources,'inventory_overlay':carry['sole_JSON_data_change'],'profile':'unchanged canonical Tests/conftest.py; no new marker/plugin/warning override','timeout_seconds':300,'private_basetemp':str(temp),'recorded_before_execution':True};(out/'diagnostic-node-argv.json').write_text(json.dumps(receipt,indent=2)+'\n');start=time.monotonic()
with (out/'diagnostic-node.stdout').open('w') as stdout,(out/'diagnostic-node.stderr').open('w') as stderr:
 p=subprocess.Popen(argv,cwd=wt,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 try:code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=124
stdout=(out/'diagnostic-node.stdout').read_text();stderr=(out/'diagnostic-node.stderr').read_text();(out/'diagnostic-node.log').write_text(stdout+'\n--- STDERR ---\n'+stderr)
import xml.etree.ElementTree as ET
xml=out/'diagnostic-node.xml';cases=list(ET.parse(xml).iter('testcase')) if xml.exists() else []
result={'argv':argv,'exit':code,'elapsed_seconds':time.monotonic()-start,'pid_and_pgid':p.pid,'process_closed':p.poll() is not None,'actual_cases':[dict(x.attrib) for x in cases],'failed_cases':[dict(x.attrib) for x in cases if x.find('failure') is not None or x.find('error') is not None],'skipped_cases':[dict(x.attrib) for x in cases if x.find('skipped') is not None],'source_after_exact':all(hashlib.sha256((wt/p).read_bytes()).hexdigest()==v for p,v in sources.items()),'head_after':subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()};(out/'diagnostic-node-result.json').write_text(json.dumps(result,indent=2)+'\n');print('diagnostic single-node exit',code,'elapsed',round(result['elapsed_seconds'],2));print(stdout);assert code==0 and len(cases)==1 and not result['failed_cases'] and not result['skipped_cases'];assert result['source_after_exact'] and result['head_after']==base
