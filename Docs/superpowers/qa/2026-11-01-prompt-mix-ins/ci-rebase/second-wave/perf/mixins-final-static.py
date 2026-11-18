import subprocess,json,collections,ast
from pathlib import Path
files=['tldw_chatbook/UI/Console_Modules/prompts.py','tldw_chatbook/Tools/character_tool_service.py','Tests/Tools/test_character_tool_service.py','tldw_chatbook/Chat/console_first_send_modules.py','Tests/Performance/test_ui_ready_module_census.py']
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';rows=[]
for p in files:
 old=subprocess.check_output(['git','show','ba1cafc55d:'+p],text=True);new=Path(p).read_text();results=[]
 for src in (old,new):
  r=subprocess.run([ruff,'check','--no-cache','--output-format=json','--stdin-filename',p,'-'],input=src,text=True,capture_output=True);results.append(collections.Counter((x['code'],x['message']) for x in json.loads(r.stdout)))
 rows.append({'file':p,'before':sum(results[0].values()),'after':sum(results[1].values()),'new':[(str(k),v) for k,v in (results[1]-results[0]).items()]})
Path('/private/tmp/mixins-final-cold-lint-comparison.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
# The approved delegates preserve the original avatar argument/default signatures.
a=ast.parse(Path('tldw_chatbook/Tools/character_tool_service.py').read_text());b=ast.parse(Path('tldw_chatbook/Character_Chat/character_avatar.py').read_text());proof=[]
for name in ('image_backend_configured','resolve_avatar'):
 x=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name);y=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name);same=ast.dump(x.args)==ast.dump(y.args) and ast.dump(x.returns)==ast.dump(y.returns);assert same;proof.append({'name':name,'exact_arguments_defaults_and_return_annotation':same})
Path('/private/tmp/mixins-final-avatar-signatures.json').write_text(json.dumps(proof,indent=2))
