from collections import Counter
from pathlib import Path
import ast
import json
import re
import subprocess

python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
base='cb4131210904f4ac4005f638eb90cc4585364a73'
paths=[p for p in subprocess.check_output(['git','diff',base,'--name-only'],text=True).splitlines() if p.endswith('.py')]
paths += [p for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines() if p.endswith('.py')]
result=[]
for name in paths:
    current=json.loads(subprocess.run([python,'-m','ruff','check','--no-cache','--config','/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook/pyproject.toml','--output-format','json',name],capture_output=True,text=True).stdout)
    original=Path('/private/tmp/mixin-task6-base')/name
    previous=json.loads(subprocess.run([python,'-m','ruff','check','--no-cache','--config','/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook/pyproject.toml','--output-format','json',str(original)],capture_output=True,text=True).stdout) if original.exists() else []
    key=lambda issue:(issue['code'],re.sub(r'from line \d+', 'from line <offset>', issue['message']))
    delta=Counter(map(key,current))-Counter(map(key,previous))
    ast.parse(Path(name).read_text(), filename=name)
    item={'file':name,'baseline':len(previous),'current':len(current),'new':list(delta.elements())}
    result.append(item)
    print(json.dumps(item))
Path('/private/tmp/mixin-task6-lint.json').write_text(json.dumps(result,indent=2))
if any(item['new'] for item in result):
    raise SystemExit(1)
