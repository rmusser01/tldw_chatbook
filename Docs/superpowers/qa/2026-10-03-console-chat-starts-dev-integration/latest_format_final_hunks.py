from pathlib import Path
import hashlib, json, os, re, subprocess, sys
scratch=Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration')
paths=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
records=[]
for path in paths:
 if not path.endswith('.py'): continue
 diff=subprocess.check_output(['git','diff','--unified=0','--',path],text=True)
 ranges=[]
 for line in diff.splitlines():
  match=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',line)
  if match:
   start=int(match[1]);size=int(match[2] or 1)
   if size:ranges.append((start,start+size-1))
 for start,end in reversed(ranges):
  argv=[sys.executable,'-m','ruff','format','--range',f'{start}-{end}',path]
  result=subprocess.run(argv,capture_output=True,text=True)
  records.append({'argv':argv,'cwd':str(Path.cwd()),'env':{k:v for k,v in os.environ.items() if k.startswith(('TLDW_','XDG_','PYTHON'))},'returncode':result.returncode,'output':result.stdout+result.stderr})
  assert result.returncode==0, records[-1]
(scratch/'latest-format-final-hunks-detail.json').write_text(json.dumps(records,indent=2))
print(f'Formatted {len(records)} modified ranges across {len(paths)} owned files.')
