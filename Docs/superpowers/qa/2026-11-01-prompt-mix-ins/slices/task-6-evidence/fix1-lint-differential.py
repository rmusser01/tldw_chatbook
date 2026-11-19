from pathlib import Path
from collections import Counter
import ast
import json
import re
import subprocess

base='817618048a994b6a8e6839f65fc81c90f33343a7'
python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
config='/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook/pyproject.toml'
root=Path('/private/tmp/mixin-task6-fix1-basefiles')
files=['tldw_chatbook/LLM_Calls/LLM_API_Calls.py','Tests/Chat/test_console_prompt_mixin_caching.py']
# Preserve exact FIX_BASE package context for module-name/import lint rules.
for name in ('tldw_chatbook/__init__.py','tldw_chatbook/LLM_Calls/__init__.py','Tests/__init__.py','Tests/Chat/__init__.py'):
    result=subprocess.run(['git','show',f'{base}:{name}'],capture_output=True)
    if result.returncode == 0:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(result.stdout)
results=[]
for name in files:
    old=root/name;old.parent.mkdir(parents=True,exist_ok=True)
    old.write_bytes(subprocess.check_output(['git','show',f'{base}:{name}']))
    def diagnostics(path):
        result=subprocess.run([python,'-m','ruff','check','--no-cache','--config',config,'--output-format','json',str(path)],capture_output=True,text=True)
        assert result.returncode in (0,1), result.stderr
        return json.loads(result.stdout)
    before=diagnostics(old);after=diagnostics(name)
    def key(issue):
        return issue['code'],re.sub(r'from line \d+', 'from line <offset>', issue['message'])
    added=Counter(map(key,after))-Counter(map(key,before))
    ast.parse(Path(name).read_text(),filename=name)
    row={'file':name,'fix_base':len(before),'current':len(after),'new':list(added.elements())}
    results.append(row);print(json.dumps(row))
Path('/private/tmp/mixin-task6-fix1-lint.json').write_text(json.dumps(results,indent=2))
assert not any(row['new'] for row in results)
