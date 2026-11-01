from pathlib import Path
import re,subprocess,sys
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c')
PYTHON='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
p=subprocess.run([PYTHON,str(ROOT/'drive.py'),'capture',sys.argv[1]],text=True,capture_output=True,check=True)
for n,line in enumerate(p.stdout.splitlines(),1):
 s=re.sub(r'\s{2,}',' ',line).strip()
 if s and set(s)-set('│─▊▔▁▎ '): print(f'{n}: {s}')
