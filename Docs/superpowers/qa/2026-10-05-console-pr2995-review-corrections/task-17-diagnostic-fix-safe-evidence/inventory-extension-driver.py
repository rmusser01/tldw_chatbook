import hashlib,json,subprocess,time
from pathlib import Path
wt=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');sdd=wt/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';out=sdd/'task-17-diagnostic-fix-safe-evidence';out.mkdir(exist_ok=True);python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python';base='408453025fc58383b81cb018b01134579cb3141a';path='Docs/security/production-diagnostic-inventory.json';report=sdd/'task-17-report-before-inventory-fix.md'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()==base
assert subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True)==''
manifest=json.loads((sdd/'task-17-safe-evidence/manifest.json').read_text());assert len(manifest['files'])==64
for name,meta in manifest['files'].items():assert hashlib.sha256((sdd/'task-17-safe-evidence'/name).read_bytes()).hexdigest()==meta['sha256']
assert hashlib.sha256(report.read_bytes()).hexdigest()==manifest['report']['sha256'];assert report.read_bytes()==(sdd/'task-17-report.md').read_bytes()
records=[]
def run(name,argv):
 start=time.monotonic();p=subprocess.run(argv,cwd=wt,text=True,capture_output=True,timeout=300);(out/f'{name}.stdout').write_text(p.stdout);(out/f'{name}.stderr').write_text(p.stderr);records.append({'name':name,'argv':argv,'exit':p.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':f'{name}.stdout','stderr':f'{name}.stderr'});(out/'mutation-receipt.json').write_text(json.dumps(records,indent=2)+'\n');print(name,p.returncode,len(p.stdout),len(p.stderr),flush=True);assert p.returncode==0;return p
r=run('diagnostic-statements-verify',[python,'scripts/check_persistent_diagnostic_inventory.py','--statements','tldw_chatbook/Chat/console_chat_start.py','--since','100fa9d819']);assert r.stdout==(sdd/'task-17-diagnostic-statements-review.txt').read_text();assert r.stderr==''
before=(wt/path).read_bytes();initial=json.loads(before)
old={'call_count':6,'diagnostic_digest':'ec602b5bd3b0fafe72cc'};new={'call_count':11,'diagnostic_digest':'568a6855dc033a46aebc'}
assert hashlib.sha256((wt/'tldw_chatbook/Chat/console_chat_start.py').read_bytes()).hexdigest()=='3d628d4394da0ecad3a4d04166623608116e709a434dab204b13ab71e6b5b2e0'
run('inventory-write',[python,'scripts/check_persistent_diagnostic_inventory.py','--write'])
after=(wt/path).read_bytes();actual=json.loads(after)
def correction(value):
 if isinstance(value,list):return [correction(x) for x in value]
 if isinstance(value,dict):
  d={k:correction(v) for k,v in value.items()}
  if d.get('path')=='tldw_chatbook/Chat/console_chat_start.py':
   assert all(d[k]==v for k,v in old.items());d.update(new)
  return d
 return value
assert correction(initial)==actual,'unexpected inventory data drift'
assert subprocess.check_output(['git','diff','--name-only'],cwd=wt,text=True).splitlines()==[path]
run('inventory-whitespace',['git','diff','--check'])
r=run('inventory-diff',['git','diff','--',path]);assert len([x for x in r.stdout.splitlines() if x.startswith('+') and not x.startswith('+++')])==2;assert len([x for x in r.stdout.splitlines() if x.startswith('-') and not x.startswith('---')])==2
(out/'inventory-only-carry.json').write_text(json.dumps({'extension_base':base,'path':path,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'sole_JSON_data_change':{'owner':'tldw_chatbook/Chat/console_chat_start.py','before':old,'after':new},'review_receipt_sha256':hashlib.sha256((sdd/'task-17-diagnostic-statements-review.txt').read_bytes()).hexdigest(),'command_handoff_row_preserved':True,'all_other_JSON_data_exact':True,'initial64_artifacts_unchanged':True,'original_manifest_report_alias':{'manifest_path':'task-17-safe-evidence/manifest.json','original_report_path':manifest['report']['path'],'frozen_report_path':str(report),'report_bytes':len(report.read_bytes()),'report_sha256':manifest['report']['sha256'],'original_manifest_sha256':hashlib.sha256((sdd/'task-17-safe-evidence/manifest.json').read_bytes()).hexdigest()}},indent=2)+'\n')
print('inventory-only exact2line correction; all64 initial artifacts exact',flush=True)
