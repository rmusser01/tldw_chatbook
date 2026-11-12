from pathlib import Path
import os,sys,json,tomllib
r=Path(__file__).parent
for k,v in {'HOME':r/'home','XDG_CONFIG_HOME':r/'xdg-config','XDG_DATA_HOME':r/'xdg-data','XDG_CACHE_HOME':r/'xdg-cache','TLDW_CONFIG_PATH':r/'config.toml'}.items():os.environ[k]=str(v)
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
sys.path.insert(0,'/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook')
from tldw_chatbook.Agents.run_hooks import load_hooks_config,RunHooksEngine
c=tomllib.loads((r/'config.toml').read_text());parsed=load_hooks_config(c)
assert parsed.enabled and len(parsed.hooks)==1
assert parsed.hooks[0].matcher=='search_library_rag'
engine=RunHooksEngine(lambda:parsed,lambda:str(r))
try:
 blocked=engine.fire('PreToolUse',session_id='guard-check',data={'tool_name':'search_library_rag','tool_args':{}})
 recovery=engine.fire('PreToolUse',session_id='guard-check',data={'tool_name':'search_archived_conversations','tool_args':{}})
 restore=engine.fire('PreToolUse',session_id='guard-check',data={'tool_name':'request_conversation_restore','tool_args':{}})
 assert blocked.blocked and not recovery.blocked and not restore.blocked
 (r/'guard-check.json').write_text(json.dumps({'parsed_hook_count':1,'rag_blocked':blocked.blocked,'reason':blocked.reason,'archive_search_hook_unblocked':not recovery.blocked,'restore_hook_unblocked':not restore.blocked,'note':'Restriction only; normal permission/runtime/confirmation gates were not bypassed.'},indent=2)+'\n')
 print('Exact production hook denies RAG; recovery calls retain normal gates.')
finally:engine.close()
