from pathlib import Path
import hashlib, json, os

ROOT = Path('/private/tmp/console-dev-live-01a0fa6c')
E = ROOT / 'evidence'
launch = json.loads((E / 'boot4-launch.json').read_text())
shutdown = json.loads((E / 'boot4-exit.json').read_text())
receipts = json.loads((E / 'boot4-final-receipts.json').read_text())
assert shutdown['returncode'] == 0 and shutdown['sha'] == launch['sha']
try:
    os.kill(shutdown['app_pid'], 0)
except ProcessLookupError:
    pass
else:
    raise AssertionError('isolated app still running')
real = Path('/Users/macbook-dev/.config/tldw_cli/config.toml')
stat = real.stat()
assert hashlib.sha256(real.read_bytes()).hexdigest() == '15c6cb224a6a51c7de5c3f716fbe9dfaef7b7ca05cf9daa7e3ebe247df9310da'
assert stat.st_size == 54970 and stat.st_mtime_ns == 1790431296768447552
conversations = [c for db in receipts['databases'].values() for c in db.get('targets', [])]
attempts = [a for db in receipts['databases'].values() for a in db.get('automatic_chat_start_attempts', [])]
chains = {c['id']: c for db in receipts['databases'].values() for c in db.get('automatic_work_chains', [])}
assert len(attempts) == 4
qualified = []
for title, token in [('DEV_FINAL_START', 'DEV_FINAL_START_OK'), ('DEV_FINAL_REMEMBERED_START', 'DEV_FINAL_REMEMBERED_START_OK')]:
    rows = [c for c in conversations if c['title'] == title]
    assert len(rows) == 1
    c = rows[0]
    handoff = json.loads(c['metadata'])['console_agent_handoff']
    aa = [a for a in attempts if a['conversation_id'] == c['id']]
    assert len(aa) == 1
    a = aa[0]
    assert (c['scope_type'], c['workspace_id']) == ('global', None)
    assert a['state'] == 'completed' and handoff['state'] == 'consumed'
    assert handoff['draft'] == '' and handoff['draft_revision'] == 2
    assert handoff['accepted_attempt_id'] == a['id'] and handoff['launch']['status'] == 'started'
    messages = c['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'user' and messages[0]['content'] == f'Reply only {token}.'
    assert messages[1]['role'] == 'assistant' and messages[1]['content'] in (token, token + '.')
    provenance = json.loads(messages[0]['metadata_json'])
    assert provenance['origin'] == 'agent_chat_start' and provenance['agent_chat_start']['attempt_id'] == a['id']
    assert not c['checkpoints']
    source, target = chains[a['source_chain_id']], chains[a['chain_id']]
    assert target['allowance_root_chain_id'] == (source['allowance_root_chain_id'] or source['id'])
    qualified.append({'title': title, 'conversation_id': c['id'], 'attempt_id': a['id'], 'reply': messages[1]['content'], 'source_chain_id': a['source_chain_id'], 'target_chain_id': a['chain_id']})
card = (E / 'boot4-remember-card-sentinel.txt').read_text()
assert all(x in card for x in ['Destination: Casual chat', 'Mode: start one bounded turn', 'Reply only DEV_FINAL_START_OK.', 'Allow for this session', 'FINAL_SOURCE_SENTINEL'])
for label, sentinel in [('boot4-first-start-result', 'FINAL_SOURCE_SENTINEL'), ('boot4-remembered-start-result', 'FINAL_REMEMBERED_SOURCE_SENTINEL')]:
    frame = (E / (label + '.txt')).read_text()
    assert sentinel in frame and 'Create chat?' not in frame
    assert 'Conversation | DEV_WORKSPACE Chat' in frame and 'Workspace    DEV_WORKSPACE' in frame
empty = (E / 'boot4-started-target-empty.txt').read_text()
assert 'Conversation | DEV_FINAL_START' in empty and 'Send disabled: type a message' in empty
edit = (E / 'boot4-target-edit-reopened.txt').read_text()
assert 'Conversation | DEV_FINAL_START' in edit and 'FINAL_LATER_TARGET_EDIT' in edit
result = {'qualified_sha': launch['sha'], 'cases': qualified, 'observed_primary_remembered_approval': True, 'started_target_composer_empty_on_open': True, 'later_target_edit_survives_navigation': True, 'app_exit_code': 0, 'app_pid_absent': True, 'real_config_sha256': hashlib.sha256(real.read_bytes()).hexdigest(), 'real_config_size_and_mtime_unchanged': True, 'limits': ['Important I2 remains open: caret/selection-only navigation before native acceptance is not qualified and can resurrect the accepted prompt.', 'The held preacceptance mounted-target interleaving and shared primary-child closure are qualified by automated regression controls, not this PTY successor.', 'No full repository suite or general descriptor/timer cleanup claim.']}
(E / 'final-live-qualification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
