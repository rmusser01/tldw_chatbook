import pathlib,ast,subprocess,json,hashlib
s=pathlib.Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge');rows=[]
for p in ['Tests/Architecture/test_persistent_diagnostic_inventory.py','tldw_chatbook/UI/Console_Modules/prompt_queue.py']:
 path=pathlib.Path(p);before=path.read_bytes();after=subprocess.check_output(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','ruff','format','--stdin-filename',p,'-'],input=before)
 assert ast.dump(ast.parse(before))==ast.dump(ast.parse(after));path.write_bytes(after);rows.append({'path':p,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'strict_ast_equal':True})
(s/'task-6-static-repair-proof.json').write_text(json.dumps(rows,indent=2)+'\n')
p=pathlib.Path('Docs/security/production-diagnostic-inventory.json');(s/'task-6-diagnostic-pin-before.json').write_bytes(p.read_bytes())
