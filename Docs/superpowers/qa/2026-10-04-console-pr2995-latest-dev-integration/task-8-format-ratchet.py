import subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts/terminal_qualification').resolve()))
import format_ratchet as ratchet
ratchet._git=lambda repo,*args:subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=repo)
ratchet._repo_root=lambda:Path.cwd().resolve()
paths=subprocess.check_output(['git','-c','gc.auto=0','diff','--name-only'],text=True).splitlines();paths=[p for p in paths if p.endswith('.py')]
out=Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-8-formatter-baseline.json')
assert out.is_file()  # Retain the original immutable baseline without rewriting it.
ratchet.verify(baseline=out,head=None)
print('Touched formatter ratchet passed for',len(paths),'paths against immutable rebase union bc184be782.')
