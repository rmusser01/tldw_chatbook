import subprocess,json,collections,ast,hashlib
from pathlib import Path
root=Path('/private/tmp/mixins-f095-firstsend');files=['tldw_chatbook/Tools/web_tool_impls.py','tldw_chatbook/Chat/console_first_send_modules.py','Tests/UI/test_console_first_send_preimport.py','Tests/Tools/test_web_deep_search.py'];rows=[];ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
for p in files:
 cs=[]
 for s in [subprocess.check_output(['git','show','f09597c:'+p],text=True),Path(p).read_text()]:
  r=subprocess.run([ruff,'check','--no-cache','--output-format=json','--stdin-filename',p,'-'],input=s,text=True,capture_output=True);cs.append(collections.Counter((x['code'],x['message']) for x in json.loads(r.stdout)))
 rows.append({'file':p,'before':sum(cs[0].values()),'after':sum(cs[1].values()),'new':[(str(k),v) for k,v in (cs[1]-cs[0]).items()],'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()})
print(json.dumps(rows,indent=2));(root/'lint-source.json').write_text(json.dumps(rows,indent=2)+'\n')
p=files[0];old=ast.parse(subprocess.check_output(['git','show','f09597c:'+p],text=True));new=ast.parse(Path(p).read_text());names={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))};proof=[]
for n in old.body:
 if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
  assert ast.dump(n,include_attributes=False)==ast.dump(names[n.name],include_attributes=False),n.name;proof.append(n.name)
(root/'unchanged-web-tool-bodies.json').write_text(json.dumps({'all_original_function_class_asts_identical':proof},indent=2)+'\n')
