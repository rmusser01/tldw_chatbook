from pathlib import Path
import json,hashlib,subprocess
qa=Path('Docs/superpowers/qa/2026-11-01-prompt-mix-ins')
def record(p):
 b=p.read_bytes();return {'sha256':hashlib.sha256(b).hexdigest(),'git_blob':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'bytes':len(b)}
old=json.loads((qa/'historical-reviewed-product-source-manifest.json').read_text())['files']
paths=set(old)|{p for p in subprocess.check_output(['git','diff','2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5','--name-only'],text=True).splitlines() if not p.startswith(str(qa)+'/')}
(qa/'product-source-manifest.json').write_text(json.dumps({'target_base':'2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5','review_fix_base':'963fe12e87ae4d2fb415bff3814943ca448cb0e0','schema_versions':{'chachanotes':86,'prompts':6},'qualification':'Current focused port plus PR3346 review-fix source and governance bytes. Live captures qualify only their recorded historical commits, not this current candidate.','files':{p:record(Path(p)) for p in sorted(paths)}},indent=2)+'\n')
(qa/'evidence-manifest.json').write_text(json.dumps({'qualification':'Preserved historical captures, port integration and current PR3346 review-fix evidence, including failures and limitations. Current product/Git source checks are strict; unavailable historical objects are explicitly unverified. Excludes only the two current manifests.','files':{str(p):record(p) for p in sorted(qa.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and p not in (qa/'evidence-manifest.json',qa/'product-source-manifest.json')}},indent=2)+'\n')
