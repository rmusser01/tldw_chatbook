from pathlib import Path
import ast,json,hashlib
p=Path('Tests/Chat/test_provider_setup_persistence.py');before=p.read_text();s=before
s=s.replace('import threading\n','import os\nimport threading\nfrom pathlib import Path\n',1)
s=s.replace('import pytest\n','import pytest\n\nfrom Tests.private_profile import private_profile_test\n',1)
names=['test_guarded_setup_rejects_completed_relevant_config_write','test_guarded_setup_rejects_completed_stored_credential_replacement','test_guarded_setup_allows_unrelated_generation_advance']
for name in names:
 start=s.index('def '+name+'(');end=s.index('\n\ndef ',start+1)
 body=s[start:end];body=body.replace('    tmp_path,\n    monkeypatch,\n','    request,\n',1).replace('config_path = tmp_path / "config.toml"','config_path = Path(os.environ["TLDW_CONFIG_PATH"])',1).replace('    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))\n','',1)
 s=s[:start]+'@pytest.mark.asyncio\n@private_profile_test\n'+body+s[end:]
def asserts(text):
 t=ast.parse(text);return {n.name:[ast.dump(x) for x in ast.walk(n) if isinstance(x,ast.Assert)] for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names}
assert asserts(before)==asserts(s)
p.write_text(s)
Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-6-provider-assertion-proof.json').write_text(json.dumps({'path':str(p),'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(s.encode()).hexdigest(),'functions':names,'original_assertions_unchanged':True,'assertions':asserts(s)},indent=2)+'\n')
