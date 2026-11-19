import ast,json,subprocess
from pathlib import Path
p=Path('Tests/Architecture/test_module_size_ratchet.py');s=p.read_text()
def rows(text):return ast.literal_eval(next(n.value for n in ast.parse(text).body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_BUDGETS'))
current=rows(s);old=rows(subprocess.check_output(['git','show','ba1cafc:Tests/Architecture/test_module_size_ratchet.py'],text=True));changed=set(subprocess.check_output(['git','diff','--name-only','ba1cafc'],text=True).splitlines())|set(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines());result={}
for path,pin in current.items():
 if path not in changed:continue
 actual=len(Path(path).read_text().splitlines())
 if path in old:assert actual<=old[path],(path,actual,old[path])
 s=s.replace(f'"{path}": {pin},',f'"{path}": {actual},',1);result[path]={'previous_ceiling':old.get(path),'current_exact_ceiling':actual}
p.write_text(s);Path('/private/tmp/mixins-root-size-pins-final.json').write_text(json.dumps(result,indent=2)+'\n');print('Exact measured rows:',len(result),'no existing ceiling raised')
