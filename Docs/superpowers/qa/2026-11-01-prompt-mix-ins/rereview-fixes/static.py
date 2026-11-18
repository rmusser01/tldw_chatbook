from pathlib import Path
import subprocess,json,collections,ast,re
base='1a58f0af2cbc5129f9c8f01595d67c330a7defda'
root=Path.cwd();private=Path('/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/prompt-mixins-pr-d18dfic_/rereview-fixes')
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
paths=[p for p in subprocess.check_output(['git','diff',base,'--name-only'],text=True).splitlines() if p.endswith('.py') and not p.startswith('Docs/')]
results={}
for path in paths:
 old=subprocess.run(['git','show',f'{base}:{path}'],capture_output=True).stdout
 def check(data):
  r=subprocess.run([ruff,'check','--output-format','json','--stdin-filename',path,'-'],input=data,capture_output=True)
  return json.loads(r.stdout)
 before=check(old) if old else [];after=check(Path(path).read_bytes());count=lambda rows:collections.Counter((r['code'],re.sub(r'from line \d+', 'from line <baseline>', r['message'])) for r in rows)
 add=count(after)-count(before)
 results[path]={'base_count':len(before),'current_count':len(after),'introduced':[{'code':c,'message':m,'count':n,'locations':[r['location'] for r in after if r['code']==c and re.sub(r'from line \d+', 'from line <baseline>', r['message'])==m]} for (c,m),n in add.items()]}
 ast.parse(Path(path).read_text())
(private/'target-lint.json').write_text(json.dumps({'base':base,'files':results},indent=2)+'\n')
print(json.dumps({p:r['introduced'] for p,r in results.items() if r['introduced']},indent=2));print('Totals',sum(r['base_count'] for r in results.values()),sum(r['current_count'] for r in results.values()))
