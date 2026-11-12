from pathlib import Path
import json,hashlib
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c');E=ROOT/'evidence'
def data(label):return json.loads((E/(label+'-receipts.json')).read_text())
def conv(d,title):
 rows=[c for db in d['databases'].values() for c in db.get('targets',[]) if c['title']==title]
 assert len(rows)==1,(title,len(rows));return rows[0]
def attempts(d):return [a for db in d['databases'].values() for a in db.get('automatic_chat_start_attempts',[])]
def handoff(c):return json.loads(c['metadata'])['console_agent_handoff']
initial=data('boot1-four-modes'); edited=data('boot1-edited-cleared'); refused=data('boot2-refused-start'); reopen=data('boot3-no-replay');manual=data('boot3-manual-recovery')
rows=[]
for title,scope,wid in [('DEV_WS_DRAFT','workspace','workspace-local-1'),('DEV_CASUAL_DRAFT','global',None)]:
 c=conv(initial,title);h=handoff(c)
 assert (c['scope_type'],c['workspace_id'])==(scope,wid)
 assert h['version']==2 and h['draft_revision']==1 and h['state']=='pending' and h['launch']['status']=='draft'
 assert not c['messages'] and not c['checkpoints'] and not [a for a in attempts(initial) if a['conversation_id']==c['id']]
 rows.append({'case':title,'result':'saved pending draft, no messages/checkpoints/start attempts','conversation_id':c['id']})
for title,scope,wid,opening,reply in [('DEV_WS_START','workspace','workspace-local-1','Reply only DEV_WS_START_OK.','DEV_WS_START_OK'),('DEV_CASUAL_START','global',None,'/settings @everyone\nReply only DEV_CASUAL_START_OK.','DEV_CASUAL_START_OK.')]:
 c=conv(initial,title);h=handoff(c);aa=[a for a in attempts(initial) if a['conversation_id']==c['id']]
 assert len(aa)==1; a=aa[0]
 assert (c['scope_type'],c['workspace_id'])==(scope,wid)
 assert a['state']=='completed' and h['state']=='consumed' and h['accepted_attempt_id']==a['id'] and h['launch']['status']=='started'
 assert [(m['role'],m['content']) for m in c['messages']]==[('user',opening),('assistant',reply)]
 provenance=json.loads(c['messages'][0]['metadata_json']);assert provenance['origin']=='agent_chat_start' and provenance['agent_chat_start']['attempt_id']==a['id']
 assert not c['checkpoints']
 chains={ch['id']:ch for db in initial['databases'].values() for ch in db.get('automatic_work_chains',[])}
 source=chains[a['source_chain_id']]; target=chains[a['chain_id']]
 assert target['allowance_root_chain_id']==(source['allowance_root_chain_id'] or source['id'])
 rows.append({'case':title,'result':'one completed native turn with machine provenance and original allowance root','conversation_id':c['id'],'attempt_id':a['id'],'source_chain_id':a['source_chain_id'],'target_chain_id':a['chain_id']})
ws=conv(edited,'DEV_WS_DRAFT');ca=conv(edited,'DEV_CASUAL_DRAFT')
assert handoff(ws)['draft']=='DEV_EDITED_WORKSPACE_DRAFT retained after restart' and handoff(ws)['draft_revision']==3
assert handoff(ca)['draft']=='' and handoff(ca)['draft_revision']==2
assert ca['system_message']=='[literal instructions] @everyone\nFollow the opening prompt and reply briefly.'
for d in (refused,reopen):
 c=conv(d,'DEV_REFUSED_START');h=handoff(c)
 assert h['launch']=={'mode':'start','status':'not_started','reason':'runtime_disabled'} and h['state']=='pending' and h['draft']=='Reply only DEV_SHOULD_NOT_AUTOSEND.'
 assert not c['messages'] and not c['checkpoints'] and not [a for a in attempts(d) if a['conversation_id']==c['id']]
 assert len(attempts(d))==2
for label,title,needle in [('boot2-edited-draft-activated','DEV_WS_DRAFT','DEV_EDITED_WORKSPACE_DRAFT retained after restart'),('boot2-cleared-draft-reopened','DEV_CASUAL_DRAFT','Send disabled: type a message'),('boot3-refused-reopened','DEV_REFUSED_START','Reply only DEV_SHOULD_NOT_AUTOSEND.')]:
 frame=(E/(label+'.txt')).read_text();assert 'Conversation | '+title in frame and needle in frame
c=conv(manual,'DEV_REFUSED_START');assert [(m['role'],m['content']) for m in c['messages']]==[('user','Reply only DEV_SHOULD_NOT_AUTOSEND.'),('assistant','DEV_SHOULD_NOT_AUTOSEND')]
assert c['messages'][0]['metadata_json'] is None and handoff(c)['state']=='consumed' and not c['checkpoints']
assert len(attempts(manual))==2 and not [a for a in attempts(manual) if a['conversation_id']==c['id']]
for label,sentinel in [('boot1-ws-draft-result','SOURCE_COMPOSER_SENTINEL'),('boot1-casual-draft-result','SOURCE_CASUAL_DRAFT_SENTINEL'),('boot1-workspace-start-result','SOURCE_WORKSPACE_START_SENTINEL'),('boot1-casual-start-result','SOURCE_CASUAL_START_SENTINEL'),('boot2-refusal-result','SOURCE_REFUSAL_SENTINEL')]:
 frame=(E/(label+'.txt')).read_text();assert 'Conversation | DEV_WORKSPACE Chat' in frame and 'Workspace    DEV_WORKSPACE' in frame and sentinel in frame
for label,needles in [('boot1-ws-draft-card-sentinel',['Destination: Workspace: workspace-local-1','Mode: save a draft','Reply only DEV_WS_DRAFT_REPLY.']),('boot1-casual-draft-card-sentinel',['Destination: Casual chat','Mode: save a draft','System prompt (explicit instructions override):','[literal instructions] @everyone','Follow the opening prompt and reply briefly.']),('boot1-workspace-start-card-sentinel',['Destination: Workspace: workspace-local-1','Mode: start one bounded turn','Reply only DEV_WS_START_OK.']),('boot1-casual-start-card-sentinel',['Destination: Casual chat','Mode: start one bounded turn','/settings @everyone','Reply only DEV_CASUAL_START_OK.']),('boot2-refusal-card-sentinel',['Destination: Casual chat','Mode: start one bounded turn','Reply only DEV_SHOULD_NOT_AUTOSEND.'])]:
 frame=(E/(label+'.txt')).read_text();assert all(x in frame for x in needles),(label,needles)
shutdown=json.loads((E/'pre-fix-shutdown-isolation.json').read_text());assert all(b['returncode']==0 and b['pid_absent'] and b['sha']=='78ff106faca1626faf74bb86029764475568df92' for b in shutdown['boots'])
assert shutdown['real_config_mtime_ns']==1790431296768447552 and shutdown['real_config_bytes']==54970 and shutdown['real_config_sha256']=='15c6cb224a6a51c7de5c3f716fbe9dfaef7b7ca05cf9daa7e3ebe247df9310da'
result={'qualified_sha':'78ff106faca1626faf74bb86029764475568df92','cases':rows,'additional_cases':['pending edit/clear revision persisted and UI reopen','automatic-disabled refusal draft, no attempt/provider send','enabled restart/reopen no automatic replay','explicit manual Send normal provenance and completed provider reply'],'clean_boots':shutdown['boots'],'real_config_unchanged':True,'limits':['Pre-fix live source, final repairs need scoped qualification.','Automated mounted controls, not PTY observations, qualify held preparation/physical drain.','First casual request staging opened Trace due missing composer focus and dispatched no case. Preserved failed automation captures.','Workspace tree first click selected a row without activating; Enter subsequently opened the intended saved draft.']}
(E/'pre-fix-live-qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
