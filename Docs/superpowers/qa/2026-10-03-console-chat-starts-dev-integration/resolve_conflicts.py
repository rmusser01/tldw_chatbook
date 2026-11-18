from pathlib import Path
import subprocess,re,json
pattern=re.compile(r'^<<<<<<<.*?\n(.*?)^=======\n(.*?)^>>>>>>>.*?\n',re.M|re.S)
choices={
'Tests/Chat/test_chat_create_confirm_card.py':['both'],
'Tests/Chat/test_console_chat_create_confirm.py':['theirs'],
'Tests/Chat/test_console_chat_create_integration.py':['theirs'],
'Tests/UI/test_console_prompt_queue.py':['theirs'],
'Tests/UI/test_console_runtime_ownership.py':['ours','ours','ours','ours','ours','theirs','theirs'],
'backlog/decisions/README.md':['both'],
'backlog/docs/lessons-backlog-hygiene.md':['both'],
'backlog/docs/lessons-testing-evidence.md':['both'],
 'tldw_chatbook/Chat/console_agent_bridge.py':['ours','ours','custom'],
 'tldw_chatbook/Chat/console_chat_controller.py':['both','custom','custom','custom','both','custom','theirs','custom','custom','custom','ours'],
 'tldw_chatbook/Chat/console_chat_store.py':['theirs'],
 'tldw_chatbook/Chat/console_dispatch_checkpoint.py':['both'],
 'tldw_chatbook/Chat/console_dispatch_repository.py':['custom'],
 'tldw_chatbook/DB/AgentRuns_DB.py':['custom','ours'],
 'tldw_chatbook/DB/ChaChaNotes_DB.py':['custom','ours'],
 'tldw_chatbook/UI/Console_Modules/prompt_queue.py':['custom'],
 'tldw_chatbook/UI/Console_Modules/workspace.py':['custom','theirs'],
 'tldw_chatbook/UI/Screens/chat_screen.py':['custom'],
}
records=[]
for name in subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],text=True).splitlines():
 p=Path(name); s=p.read_text(); n=0
 if not pattern.search(s):continue
 def resolve(m):
  global n
  i=n;n+=1; a,b=m[1],m[2]; kind=choices[name][i]
  records.append({'file':name,'block':i,'choice':kind,'ours':a,'feature':b})
  if kind=='ours':return a
  if kind=='theirs':return b
  if kind=='both':return a+b
  if name.endswith('console_agent_bridge.py'):
   return b[:b.index('        if grant_scope not in remembered:')]+a.replace('            denials[tool] += 1\n','            denials[tool] += 1\n            _release_chat_creation_token(payload)\n')
  if name.endswith('console_chat_controller.py'):
   if i==1:return a.replace('elif origin is ConsoleSubmissionOrigin.AGENT_WAKE:', 'elif origin in {ConsoleSubmissionOrigin.AGENT_WAKE, ConsoleSubmissionOrigin.AGENT_CHAT_START}:')
   if i==2:return b+'            and machine_input is None\n'
   if i==3:
    metadata=b[b.index('                    MessageMetadata('):b.index('                persist=False,')]
    metadata=metadata.replace('                    else None','                    else MessageMetadata(origin=MESSAGE_ORIGIN_HOOK)\n                    if machine_input is not None\n                    else None')
    return a[:a.index('                        MessageMetadata(')]+metadata+a[a.index('                    attachments=staged_attachments,'):]
   if i==5:return b+'                and not continuation.hook_continuation\n'
   if i==7:return '        from tldw_chatbook.Agents.agent_models import AGENT_KIND_PRIMARY\n'
   if i==8:return b.replace('        if grant in self._chat_create_session_grants.get(owning_session_id, set()):','        if requesting_kind == AGENT_KIND_PRIMARY and grant in self._chat_create_session_grants.get(owning_session_id, set()):')
   if i==9:return a+'            return True\n'+b
  if name.endswith('console_dispatch_repository.py'):return b+a
  if name.endswith('AgentRuns_DB.py'):return a.replace('_CURRENT_SCHEMA_VERSION = 21','_CURRENT_SCHEMA_VERSION = 22')
  if name.endswith('ChaChaNotes_DB.py'):return '    _CURRENT_SCHEMA_VERSION = 76  # Native agent-chat-start dispatch receipts.\n'
  if name.endswith('prompt_queue.py'):return a.replace('        return derive_prompt_queue_presentation(', '        presentation = derive_prompt_queue_presentation(')
  if name.endswith('workspace.py'):return a+'import asyncio\nfrom datetime import datetime, timezone\nimport inspect\nimport json\nimport re\nimport time\n'
  if name.endswith('chat_screen.py'):return b.replace('                    active_leaf_persisted_id=active_leaf_persisted_id,','                    active_leaf_persisted_id=active_leaf_persisted_id,\n                    settings=settings,').replace('                if opening_prompt:', '                if opening_prompt and not session.draft:')
  raise RuntimeError((name,i))
 result=pattern.sub(resolve,s); assert n==len(choices[name]),(name,n,len(choices[name]));p.write_text(result)
Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/conflict-dispositions.json').write_text(json.dumps(records,indent=2))
print('Resolved',len(records),'blocks')
