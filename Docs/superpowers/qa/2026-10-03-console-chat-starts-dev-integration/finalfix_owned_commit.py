from pathlib import Path
import json,subprocess,sys
root=Path(__file__).resolve().parent
paths=['Tests/Chat/test_console_chat_start.py', 'Tests/UI/test_console_session_controller.py', 'Tests/Chat/test_console_chat_create_integration.py', 'Tests/UI/test_console_runtime_ownership.py', 'tldw_chatbook/Chat/console_agent_bridge.py', 'tldw_chatbook/Chat/console_chat_controller.py', 'tldw_chatbook/Chat/console_chat_store.py', 'tldw_chatbook/UI/Console_Modules/session.py']
def run(label,argv):
 r=subprocess.run([sys.executable,'-B',str(root/'run_check.py'),label,*argv]);assert r.returncode==0,(label,r.returncode)
 return json.loads((root/(label+'.json')).read_text())['output']
assert run('finalfix-precommit-head',['git','rev-parse','HEAD']).strip()=='473c7b26eccf1e7b77c4c8e2d4c7f57ccb840afe'
assert run('finalfix-precommit-index',['git','diff','--cached','--name-only']).splitlines()==[]
assert sorted(run('finalfix-precommit-owned-diff',['git','diff','--name-only']).splitlines())==sorted(paths)
run('finalfix-owned-add',['git','-c','gc.auto=0','add','--',*paths])
assert sorted(run('finalfix-owned-stage-proof',['git','diff','--cached','--name-only']).splitlines())==sorted(paths)
run('finalfix-owned-stage-whitespace',['git','diff','--cached','--check','--',*paths])
run('finalfix-owned-commit',['git','-c','gc.auto=0','commit','-m','fix(console): preserve child approval and consume mounted handoffs'])
run('finalfix-owned-commit-head',['git','rev-parse','HEAD'])
run('finalfix-owned-final-tracked-status',['git','status','--short','--untracked-files=no'])
