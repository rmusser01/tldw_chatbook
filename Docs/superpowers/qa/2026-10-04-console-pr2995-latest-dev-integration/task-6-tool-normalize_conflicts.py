import ast,json,subprocess,pathlib,hashlib
root=pathlib.Path.cwd(); scratch=pathlib.Path('/private/tmp/pr2995-task6')
sha=subprocess.check_output(['git','rev-parse','REBASE_HEAD'],text=True).strip()
paths=subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],text=True).splitlines()
records=[]
for p in paths:
 if not p.endswith('.py'): continue
 files=[];record={'commit':sha,'path':p,'stages':[]}
 for n in [2,1,3]:
  raw=subprocess.check_output(['git','show',f':{n}:{p}'])
  formatted=subprocess.run(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','ruff','format','--stdin-filename',p,'-'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
  assert ast.dump(ast.parse(raw))==ast.dump(ast.parse(formatted)), (p,n,'AST changed')
  f=scratch/(sha[:10]+'-'+p.replace('/','--')+f'.{n}');f.write_bytes(formatted);files.append(str(f))
  record['stages'].append({'stage':n,'raw_sha256':hashlib.sha256(raw).hexdigest(),'formatted_sha256':hashlib.sha256(formatted).hexdigest(),'ast_equal':True})
 merged=subprocess.run(['git','merge-file','-p',*files],stdout=subprocess.PIPE)
 out=scratch/(sha[:10]+'-'+p.replace('/','--')+'.merged');out.write_bytes(merged.stdout)
 record['merge_exit']=merged.returncode
 if merged.returncode==0:
  ast.parse(merged.stdout);(root/p).write_bytes(merged.stdout);subprocess.run(['git','add','--',p],check=True)
 records.append(record)
with (scratch/'conflict-records.jsonl').open('a') as f:
 for record in records:f.write(json.dumps(record)+'\n')
print(json.dumps(records,indent=2))
