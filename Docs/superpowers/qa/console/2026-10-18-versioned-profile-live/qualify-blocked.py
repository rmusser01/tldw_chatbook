"""Assert only observed facts of the bounded, incomplete native phase."""
from pathlib import Path
import json

r=Path(__file__).parent
case=r/'direct-ready'
load=lambda name:json.loads((case/(name+'.json')).read_text())
b,s,f=map(load,('baseline','after-search','final'))
for name in ('conversations','messages','workspace_records'):
    assert b[name]==s[name]==f[name],name
assert b['conversation_count']==3 and s['conversation_count']==f['conversation_count']==4
for name,key in [('calls','call_id'),('reconstructions','call_id'),('responses','call_id'),('thinking_structure','message_id')]:
    final={item[key]:item for item in f[name]}
    assert all(final[item[key]]==item for item in s[name]),name
assert len(s['calls'])==2 and len(f['calls'])==5
assert all(item['state']=='complete' for item in f['calls'])
assert s['responses'][-1]['link']['link_kind']=='revision'
assert s['responses'][-1]['link']['verification_outcome']=='verified_equal'
assert all(item['profile']['version']==2 for item in f['responses'])
assert all(item['profile']['source_formats']==['start_anchored_think','reasoning_content'] for item in f['responses'])
assert s['thinking_structure'][0]['blocks'][-1]['source_format']=='reasoning_content'
users=[m for m in f['source_messages'] if m['role']=='user']
assert [m['content'] for m in users]==['Find recently archived chats across all workspaces.','Restore the second one.']
assert len({m['conversation_id'] for m in users})==1
for request in f['reconstructions']:
    own=[m for m in f['user_revisions'] if m['source_message_id']==request['turn_id']]
    assert len(own)==1 and request['boundary_source_revisions']==[own[0]['revision_id']]
    assert not request['omissions'] and request['archive_handoffs']==0
rows=[json.loads(line) for line in (r/'transport-metadata.jsonl').read_text().splitlines()]
requests=[x for x in rows if x['event']=='request']
http=[x for x in rows if x['event']=='http']
complete=[x for x in rows if x['event']=='response_observation']
assert [x['attempt'] for x in requests]==list(range(18,23))
assert len(http)==len(complete)==5 and all(x['status']==200 for x in http)
assert all(x['sse_done'] and not x['invalid_json'] and not x['observation_truncated'] for x in complete)
assert all(x['retry_total']==0 and x['transport']=='httpx_async' and x['tool_names']==[] for x in requests)
assert not any(x['event'] in {'transport_exception','denied_cap'} for x in rows)
assert json.loads((r/'counter.json').read_text())=={'consumed':22,'authorized_total':30}
assert json.loads((r/'normal-before.json').read_text())==json.loads((r/'normal-after.json').read_text())
d=json.loads((r/'capability-diagnosis.json').read_text())
assert d['ready'] and not d['native_tools_supported'] and d['model_requests']==0 and d['counter_unchanged']
assert any(x['path']=='/props' and x['status']==404 for x in d['get_observations'])
result={'complete_provider_calls':5,'model_attempts':[x['attempt'] for x in requests],'http200_sse_done':5,'terminal_search_response_verified_equal':True,'v2_literal_structured_provenance':True,'original_rows_messages_workspaces_unchanged':True,'predecessor_calls_headers_links_reconstructions_text_thinking_exact':True,'exact_current_user_source_pins':True,'followup_handoffs':0,'actual_native_tools_supported':False,'optional_rag_capture_failure':'co-occurs; not diagnosed or claimed causal','restore_confirmation_restore_open':'not reached','fresh_rag_only':'not launched after the same protocol gate was established','counter':22,'cap':30,'remaining':8,'normal_fingerprints_equal':True}
(case/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
