"""Assert read-only observations of the actual synthetic native recovery flow."""
import json
from pathlib import Path
import sys

root = Path(__file__).parent
case = root / sys.argv[1]
load = lambda phase: json.loads((case / (phase + '.json')).read_text())
first, pending, restored, final = map(load, ('after-search', 'pending', 'restored', 'final'))
assert len(first['calls']) == 2 and all(x['state'] == 'complete' for x in first['calls'])
assert len(pending['calls']) == 3
assert len(restored['calls']) == len(final['calls']) == 4
assert all(x['state'] == 'complete' for x in restored['calls'] + final['calls'])
for phase in (pending, restored, final):
    assert phase['conversations'] == first['conversations']
    assert phase['messages'] == first['messages']
    assert phase['conversation_count'] == first['conversation_count'] == 4
    calls = {x['call_id']: x for x in phase['calls']}
    requests = {x['call_id']: x for x in phase['reconstructions']}
    for old in first['calls']:
        assert calls[old['call_id']] == old
    for old in first['reconstructions']:
        assert all(requests[old['call_id']][key] == value for key, value in old.items())
assert pending['workspace_records'] == first['workspace_records']
before_ws = {x['workspace_id']: x for x in first['workspace_records']}
after_ws = {x['workspace_id']: x for x in restored['workspace_records']}
assert set(before_ws) == set(after_ws)
for key, old in before_ws.items():
    current = after_ws[key]
    if key != 'atlas':
        assert current == old
        continue
    assert old['archived'] == 1 and current['archived'] == 0
    assert current['archive_revision'] == old['archive_revision'] + 1
    assert current['archived_at'] is None
    assert {k for k in old if old[k] != current[k]} == {'archived', 'archived_at', 'archive_revision', 'updated_at'}
users = [x for x in final['source_messages'] if x['role'] == 'user']
assert [x['content'] for x in users] == ['Find recently archived chats across all workspaces.', 'Restore the second one.']
assert len({x['conversation_id'] for x in users}) == 1
for request in final['reconstructions']:
    own = [x for x in final['user_revisions'] if x['source_message_id'] == request['turn_id']]
    assert len(own) == 1 and request['boundary_source_revisions'] == [own[0]['revision_id']]
    assert not request['omissions']
assert sorted(x['archive_handoffs'] for x in final['reconstructions']) == [0, 0, 1, 1]
wire = [json.loads(s) for s in (case / 'wire-metadata.jsonl').read_text().splitlines()]
requests = [x for x in wire if x['event'] == 'request']
http = [x for x in wire if x['event'] == 'http']
ends = [x for x in wire if x['event'] == 'completion']
assert len(requests) == len(http) == len(ends) == 4
assert all(x['status'] == 200 for x in http)
assert all(x['done'] and not x['sse_error'] and not x['invalid_json_lines'] and 'exception_type' not in x for x in ends)
assert all(x['transport_retry_total'] == 0 for x in requests)
assert [x['tool_names'] for x in ends] == [['search_archived_conversations'], [], ['request_conversation_restore'], []]
assert not any(x['event'] in ('transport_exception', 'http_error') for x in wire)
result = {'mode': sys.argv[1], 'model_attempts': [x['attempt'] for x in requests], 'http_200_sse_complete': 4, 'complete_capture_calls': 4, 'original_fixture_rows_and_messages_unchanged': True, 'only_selected_workspace_archive_state_changed_before_open': True, 'exact_current_user_call_boundary_pins': True, 'predecessor_call_rows_and_reconstruction_unchanged': True, 'saved_user_texts_exact': True, 'no_duplicate_conversation': True, 'source_unsent_draft_verified': 'native UI evidence'}
(case / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
