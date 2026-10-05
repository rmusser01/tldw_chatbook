import re, subprocess
path='Tests/UI/test_console_runtime_ownership.py'
diff=subprocess.check_output(['git','diff','--unified=0','c0b251a71422536a3da2be421dd60f79e2cb230a','--',path],text=True)
ranges=[]
for line in diff.splitlines():
 m=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',line)
 if m:
  start=int(m[1]);count=int(m[2] or '1')
  if count:ranges.append((start,start+count))
for start,end in reversed(ranges):
 print('format range',start,end,flush=True)
 subprocess.run(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','ruff','format','--range',f'{start}-{end}',path],check=True)
