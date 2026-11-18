import ast,collections,hashlib,json,subprocess
from pathlib import Path
base='c0b251a71422536a3da2be421dd60f79e2cb230a'
path='Tests/UI/test_console_runtime_ownership.py'
original=ast.parse(subprocess.check_output(['git','show',f'{base}:{path}'],text=True))
current=ast.parse(Path(path).read_text())
functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old,new=functions(original),functions(current)
assert set(old)<=set(new)
results={}
for name,n in old.items():
 before=[ast.dump(x,include_attributes=False) for x in ast.walk(n) if isinstance(x,ast.Assert)]
 after=[ast.dump(x,include_attributes=False) for x in ast.walk(new[name]) if isinstance(x,ast.Assert)]
 preserved=not (collections.Counter(before)-collections.Counter(after))
 assert preserved,name
 assert [ast.dump(x) for x in n.decorator_list]==[ast.dump(x) for x in new[name].decorator_list],name
 results[name]={'original_assertions':len(before),'final_assertions':len(after),'all_original_assertions_preserved':preserved,'decorators_identical':True}
prod=['tldw_chatbook/app.py','tldw_chatbook/app_navigation.py','tldw_chatbook/UI/Screens/chat_screen.py','tldw_chatbook/Chat/console_runtime.py','tldw_chatbook/UI/Console_Modules/session.py','tldw_chatbook/Chat/console_chat_controller.py','tldw_chatbook/Chat/console_chat_start.py','tldw_chatbook/DB/automatic_work.py']
assert subprocess.check_output(['git','diff',base,'--',*prod])==b''
report={'base':base,'file':path,'final_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'original_function_count':len(old),'new_functions':sorted(set(new)-set(old)),'assertions':results,'production_diff_empty':True}
Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-5-evidence/self-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: {len(old)} original function assertion/decorator sets preserved; no production changes; '+report['final_sha256'])
