import ast,subprocess,json,collections
from pathlib import Path
files=['tldw_chatbook/Widgets/Console/console_composer_bar.py','tldw_chatbook/Widgets/Console/console_speech_controls.py','tldw_chatbook/LLM_Calls/LLM_API_Calls.py','tldw_chatbook/LLM_Calls/openai_protocol.py','tldw_chatbook/UI/Console_Modules/composer_run_controls.py','Tests/Chat/test_openai_streaming_usage.py','Tests/UI/test_console_voice_chip.py','Tests/UI/test_console_idle_voice_boot.py','tldw_chatbook/Chat/console_first_send_modules.py','tldw_chatbook/app_idle_warmup.py','Tests/UI/test_console_first_send_preimport.py','Tests/UI/test_console_idle_heap_refreeze.py']
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';oldall=collections.Counter();newall=collections.Counter();rows=[]
for p in files:
 old=subprocess.run(['git','show','ba1cafc55d:'+p],text=True,capture_output=True).stdout;new=Path(p).read_text();cs=[]
 for source in (old,new):
  r=subprocess.run([ruff,'check','--no-cache','--output-format=json','--stdin-filename',p,'-'],input=source,text=True,capture_output=True);cs.append(collections.Counter((x['code'],x['message']) for x in json.loads(r.stdout)))
 oldall.update(cs[0]);newall.update(cs[1]);rows.append({'path':p,'old':sum(cs[0].values()),'new':sum(cs[1].values()),'added':[(str(k),v) for k,v in (cs[1]-cs[0]).items()]})
out={'files':rows,'old_total':sum(oldall.values()),'new_total':sum(newall.values()),'introduced':[(str(k),v) for k,v in (newall-oldall).items()]}
Path('/private/tmp/mixins-extraction-lint.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
a=json.loads(Path('/private/tmp/mixins-extraction-styles-baseline.json').read_text());b=json.loads(Path('/private/tmp/mixins-extraction-styles-after.json').read_text());shared=a.keys()&b.keys();diff=[{'key':json.loads(k),'fields':[f for f in a[k] if a[k][f]!=b[k][f]]} for k in sorted(shared) if a[k]!=b[k]];out={'baseline_records':len(a),'after_records':len(b),'shared':len(shared),'identical':len(shared)-len(diff),'differences':diff};Path('/private/tmp/mixins-extraction-style-comparison.json').write_text(json.dumps(out,indent=2));print('STYLE',len(shared),len(diff));print('DIFFEXAMPLES',diff[:3])
