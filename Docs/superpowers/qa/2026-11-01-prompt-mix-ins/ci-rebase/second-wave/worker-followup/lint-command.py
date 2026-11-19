import collections,difflib,hashlib,json,pathlib,re,subprocess
p=pathlib.Path('/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-worker-final-z7pf1nj1'); prior=json.loads(pathlib.Path('/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci3346-core-extract-i4xaa7gn/lint.json').read_text());exe='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';out={}
for f in ['tldw_chatbook/Chat/console_chat_controller.py','tldw_chatbook/Chat/console_agent_reply.py','Tests/Scripts/test_check_textual_worker_contract.py']:
 s=pathlib.Path(f).read_text()
 def lint(src):return json.loads(subprocess.run([exe,'check','--no-cache','--output-format','json','--stdin-filename',f,'-'],input=src,text=True,capture_output=True).stdout)
 old=prior[f]['after']['lint'] if f in prior else lint(subprocess.check_output(['git','show','HEAD:'+f],text=True));now=lint(s);key=lambda rows:collections.Counter((r['code'],re.sub(r'from line \d+','from line <location>',r['message'])) for r in rows)
 out[f]={'sha256':hashlib.sha256(s.encode()).hexdigest(),'lines':len(s.splitlines()),'new_lint':list((key(now)-key(old)).elements()),'current_lint':now}
 print(f,'lines',len(s.splitlines()),'new_lint',out[f]['new_lint'])
(p/'lint.json').write_text(json.dumps(out,indent=2)+'\n')
