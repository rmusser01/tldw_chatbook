import ast,json,site,subprocess,sys
from pathlib import Path
src=Path('tldw_chatbook/Chat/console_first_send_modules.py');tree=ast.parse(src.read_text());inventory=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='FIRST_SEND_MODULES'));ordered=[n for n in inventory if n in {'six','webencodings','html5lib'}]
script='''import importlib,json,sys
sys.path.extend(json.loads(sys.argv[1]))
assert "six" not in sys.modules
rows=[]
for name in json.loads(sys.argv[2]):
 before={n.split(".")[0] for n in sys.modules}
 importlib.import_module(name)
 loaded={n.split(".")[0] for n in sys.modules}-before
 rows.append({"step":name,"relevant_packages":sorted(loaded & {"six","webencodings","html5lib"})})
print(json.dumps(rows))
'''
rows={}
for label,steps in [('previous',['webencodings','html5lib']),('current',ordered)]:
 p=subprocess.run([sys.executable,'-S','-c',script,json.dumps(site.getsitepackages()),json.dumps(steps)],capture_output=True,text=True,check=True);rows[label]=json.loads(p.stdout)
assert rows['previous'][-1]['relevant_packages']==['html5lib','six']
assert all(row['relevant_packages']==[row['step']] for row in rows['current'])
Path('/private/tmp/mixins-f095-firstsend/lean-html-order.json').write_text(json.dumps({'method':'fresh -S interpreter, explicit existing venv site path, no app imports or sys.modules removal','ordered_current_inventory':ordered,'observations':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
