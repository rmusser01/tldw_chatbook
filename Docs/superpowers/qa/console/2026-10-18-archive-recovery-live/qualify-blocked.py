"""Verify retained read-only facts of the bounded, unsuccessful native phase."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tomllib

root = Path(__file__).parent
case = root / 'direct-bare-persona'
phases = {name: json.loads((case / (name + '.json')).read_text()) for name in ('baseline', 'after-search', 'pending', 'final')}
first, pending, final = (phases[x] for x in ('after-search', 'pending', 'final'))
assert len(first['calls']) == 2
assert {row['route_identity'] for row in first['calls']} == {'agent_first', 'tool_loop'}
assert all(row['state'] == 'complete' and row['outcome'] == 'complete' for row in first['calls'])
for observed in (pending, final):
    assert observed['calls'] == first['calls']
    assert observed['reconstructions'] == first['reconstructions']
    assert observed['conversations'] == phases['baseline']['conversations']
    assert observed['messages'] == phases['baseline']['messages']
    assert observed['workspace_records'] == first['workspace_records']
    assert observed['conversation_count'] == 4
    assert all(row in observed['source_messages'] for row in first['source_messages'])
users = [row for row in final['source_messages'] if row['role'] == 'user']
assert sorted(row['content'] for row in users) == sorted(['Find recently archived chats across all workspaces.', 'Restore the second one.'])
assert len({row['conversation_id'] for row in users}) == 1
for request in final['reconstructions']:
    owners = [row for row in final['user_revisions'] if row['source_message_id'] == request['turn_id']]
    assert len(owners) == 1 and request['boundary_source_revisions'] == [owners[0]['revision_id']]
    assert not request['omissions'] and request['archive_handoffs'] == 0
assert not any(row['content'] == 'Keep this source draft.' for row in final['source_messages'])
fixture = json.loads((case / 'fixture.json').read_text())
with sqlite3.connect('file:' + fixture['paths']['chat_db'] + '?mode=ro', uri=True) as db:
    thinking = [json.loads(row[0]) for row in db.execute("SELECT thinking_blocks_json FROM messages WHERE conversation_id=? AND thinking_blocks_json IS NOT NULL", (users[0]['conversation_id'],))]
    headers = [dict(zip(('route', 'generation'), row)) for row in db.execute('SELECT route_identity,generation_parameters_json FROM console_trace_request_headers ORDER BY created_at,header_id')]
assert len(thinking) == 1 and len(thinking[0]['blocks']) == 2
assert all(row['route'] != 'llama_fallback' for row in headers)
assert not any(json.loads(line)['event'] == 'request' for line in (case / 'wire-metadata.jsonl').read_text().splitlines())
config = tomllib.loads((case / 'config.toml').read_text())
assert config['console']['agent_max_model_turns'] == 2
assert config['console']['agent_max_model_retries'] == 0
log = (case / 'data/task7_live/tldw_cli_app.log').read_text()
assert 'exception_type=ValueError phase=trace_reservation' in log
assert log.count('phase=provider_entry status=entered') == 2
assert 'Trace reservation status is unavailable.' in (case / 'canceled.txt').read_text()
assert 'Restore the second one.' in (case / 'followup-draft.txt').read_text()
assert 'Linden planning' in (case / 'findings-ordered.txt').read_text()
normal = {}
for name in json.loads((root / 'normal-before.json').read_text()):
    p = Path(name)
    normal[name] = {'exists': p.exists()}
    if p.exists():
        body = p.read_bytes()
        normal[name].update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
assert normal == json.loads((root / 'normal-before.json').read_text())
(root / 'normal-after.json').write_text(json.dumps(normal, indent=2) + '\n')
result = {'outcome': 'Direct restore follow-up blocked before successor reservation/transport; not qualified', 'search_complete_capture_calls': 2, 'call_routes': sorted(row['route_identity'] for row in first['calls']), 'live_historical_thinking_blocks': 2, 'wire_observer_limit': 'requests observer did not intercept native HTTPX; no HTTP status/SSE claims', 'request_accounting': {'previous_used': 15, 'native_provider_entries_this_phase': 2, 'used_total': 17, 'remaining_of_20': 3, 'basis': 'two complete durable call admissions/provider entries, no fallback route; not wire events'}, 'fixture_rows_messages_unchanged': True, 'predecessor_rows_reconstructions_unchanged': True, 'exact_search_current_user_source_pins': True, 'no_restore_or_open': True, 'cancel_refused': True, 'native_quit_pty_exit': 0, 'normal_profile_unchanged': True}
(case / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
