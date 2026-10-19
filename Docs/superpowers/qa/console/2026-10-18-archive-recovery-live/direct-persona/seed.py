import json,os,pathlib,sys
R=pathlib.Path(__file__).parent
for name,path in {'HOME':R/'home','XDG_CONFIG_HOME':R/'xdg-config','XDG_DATA_HOME':R/'xdg-data','XDG_CACHE_HOME':R/'xdg-cache','TLDW_CONFIG_PATH':R/'config.toml'}.items():os.environ[name]=str(path)
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
sys.path.insert(0,'/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook')
from tldw_chatbook import config
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
assert str(config.get_user_data_dir()).startswith(str(R))
chat=CharactersRAGDB(config.get_chachanotes_db_path(),client_id='task7-live')
wdb=WorkspaceDB(config.get_workspaces_db_path())
reg=LocalWorkspaceRegistryService(wdb)
reg.create_workspace(workspace_id='atlas',name='Atlas Research')
reg.create_workspace(workspace_id='vault',name='Vault Trips')
cases=[]
for key,title,workspace,chat_archive,ws_archive in [
 ('chat_only','Maple archival notes',None,True,False),
 ('workspace_only','Linden planning conversation','atlas',False,True),
 ('both','Birch itinerary conversation','vault',True,True),
]:
 payload={'title':title}
 if workspace:payload.update(scope_type='workspace',workspace_id=workspace)
 cid=chat.add_conversation(payload)
 assert cid
 mid=chat.add_message({'conversation_id':cid,'sender':'user','content':f'Original history for {title}'})
 leaf=chat.add_message({'conversation_id':cid,'sender':'assistant','content':f'Saved reply for {title}','parent_message_id':mid})
 chat.set_conversation_active_leaf(cid,leaf)
 if chat_archive:chat.set_conversations_archived([cid],archived=True,expected_versions={cid:1})
 cases.append({'key':key,'conversation_id':cid,'title':title,'workspace_id':workspace,'message_ids':[mid,leaf],'active_leaf':leaf,'chat_archived':chat_archive,'workspace_archived':ws_archive})
reg.archive_workspace('atlas')
reg.archive_workspace('vault')
from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
personas=LocalCharacterPersonaService(chat,persona_store_path=config.get_user_data_dir()/'tldw_chatbook_personas.json')
persona=personas.create_persona_profile({'id':'local-persona-archive-recovery-check','name':'Archive Recovery Assistant','system_prompt':'Help the user find archived chats and choose whether to restore one. Present the findings before requesting restoration.','use_persona_state_context_default':False,'policy_rules':[{'rule_kind':'mcp_tool','rule_name':name,'allowed':True} for name in ('search_archived_conversations','request_conversation_restore')]})
reg.create_workspace(workspace_id='recovery-check',name='Recovery Check',assistant_defaults=WorkspaceAssistantDefaults(assistant_id=persona['id']))
(R/'persona-control.json').write_text(json.dumps({'persona':persona,'workspace_id':'recovery-check','note':'Existing product narrowing policy, no tool or confirmation gate bypass.'},indent=2)+'\n')
assert not any(config.get_user_data_dir().glob('*rag_index*'))
(R/'fixture.json').write_text(json.dumps({'cases':cases,'paths':{'user_data':str(config.get_user_data_dir()),'chat_db':str(config.get_chachanotes_db_path()),'workspace_db':str(config.get_workspaces_db_path())}},indent=2)+'\n')
print(json.dumps({'cases':cases,'no_rag_index':True}))
wdb.close()
with chat.quiesce_connections(timeout_seconds=2.0):pass
