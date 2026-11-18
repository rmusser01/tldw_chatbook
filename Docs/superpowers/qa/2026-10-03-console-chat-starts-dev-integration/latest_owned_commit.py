from pathlib import Path
import json,subprocess,sys
root=Path(__file__).resolve().parent
paths=['Tests/Chat/test_console_provider_gateway.py','Tests/UI/test_console_cost_chip_screen.py','Tests/UI/test_console_spend_projection.py','Tests/Utils/test_egress.py','tldw_chatbook/Chat/console_runtime.py','tldw_chatbook/UI/Screens/chat_screen.py']
def run(label,argv):
 r=subprocess.run([sys.executable,'-B',str(root/'run_check.py'),label,*argv])
 assert r.returncode==0,(label,r.returncode)
 return json.loads((root/(label+'.json')).read_text())['output']
assert run('latest-precommit-head',['git','rev-parse','HEAD']).strip()=='62806f75a1ad5e293e25a00123278c5f1307d601'
assert run('latest-precommit-index',['git','diff','--cached','--name-only']).splitlines()==[]
assert sorted(run('latest-precommit-owned-diff',['git','diff','--name-only']).splitlines())==sorted(paths)
run('latest-owned-add',['git','-c','gc.auto=0','add','--',*paths])
assert sorted(run('latest-owned-stage-proof',['git','diff','--cached','--name-only']).splitlines())==sorted(paths)
run('latest-owned-stage-whitespace',['git','diff','--cached','--check','--',*paths])
run('latest-owned-commit',['git','-c','gc.auto=0','commit','-m','fix(console): cancel idle spend refresh under turn custody'])
run('latest-owned-commit-head',['git','rev-parse','HEAD'])
run('latest-owned-final-status',['git','status','--short'])
