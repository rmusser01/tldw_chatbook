from pathlib import Path
import re,subprocess,sys
r=Path(__file__).parent
result=subprocess.run(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python',str(r/'drive.py'),'cap',sys.argv[1]],capture_output=True,text=True,check=True)
for n,line in enumerate(result.stdout.splitlines(),1):
 if line.strip() and (n<=8 or n>=42 or re.search(r'archive|restore|preview|confirm|cancel|scope|workspace|library|auto |agent |provider|model|birch|linden|maple|receipt|failed|error|limited',line,re.I)):
  print(f'{n}: {line}')
