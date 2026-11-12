from pathlib import Path
import subprocess, re, json, collections
root=Path.cwd(); base='b69e9114626e21292104240cc70e912485a90c54'
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
paths=[p for p in subprocess.check_output(['git','diff','--name-only',base],text=True).splitlines() if p.endswith('.py')]
paths.append('Tests/Chat/test_console_prompt_mixin_forks.py')
for path in paths:
    old=subprocess.run(['git','show',f'{base}:{path}'],capture_output=True).stdout
    if not old:
        subprocess.run([ruff,'format',path],check=True,capture_output=True)
        continue
    diff=subprocess.check_output(['git','diff','--unified=0',base,'--',path],text=True)
    ranges=[(int(a),int(b or '1')) for a,b in re.findall(r'^@@.*\+(\d+)(?:,(\d+))? @@',diff,re.M)]
    for start,count in reversed(ranges):
        if count: subprocess.run([ruff,'format','--range',f'{start}-{start+count}',path],check=True,capture_output=True)
results={}
for path in paths:
    old=subprocess.run(['git','show',f'{base}:{path}'],capture_output=True).stdout
    def check(data):
        result=subprocess.run([ruff,'check','--output-format','json','--stdin-filename',path,'-'],input=data,capture_output=True)
        return json.loads(result.stdout)
    before=check(old) if old else []; after=check(Path(path).read_bytes())
    counts=lambda rows:collections.Counter((row['code'],row['message']) for row in rows)
    added=counts(after)-counts(before)
    results[path]={'base_count':len(before),'current_count':len(after),'introduced':[
        {'code':code,'message':message,'count':n,'locations':[row['location'] for row in after if row['code']==code and row['message']==message]}
        for (code,message),n in added.items()]}
Path('/private/tmp/mixins-final-lint.json').write_text(json.dumps({'base':base,'files':results},indent=2)+'\n')
print(json.dumps({p:r for p,r in results.items() if r['introduced']},indent=2))
print('Totals:',sum(x['base_count'] for x in results.values()),sum(x['current_count'] for x in results.values()))
