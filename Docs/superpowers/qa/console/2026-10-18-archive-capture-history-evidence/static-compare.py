import ast
import collections
import hashlib
import json
import subprocess
from pathlib import Path

py = '/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
base_ref = '75b671c001e84bf2713c380e30de3e587bcc8ae0'
qa = Path('Docs/superpowers/qa/console/2026-10-18-archive-capture-history-evidence')
paths = [
    'tldw_chatbook/Chat/console_chat_controller.py',
    'tldw_chatbook/Chat/console_agent_bridge.py',
    'tldw_chatbook/Chat/console_trace_service.py',
    'Tests/Chat/test_console_archive_recovery_followup.py',
    'Tests/Chat/test_console_trace_service.py',
]
summary = {}
for name in paths:
    baseline = subprocess.run(['git', 'show', base_ref + ':' + name], check=True, stdout=subprocess.PIPE).stdout
    current = Path(name).read_bytes()
    entries = []
    for phase, content in [('baseline', baseline), ('current', current)]:
        label = phase + '-' + Path(name).stem
        command = [py, '-m', 'ruff', 'check', '--no-cache', '--output-format=json', '--stdin-filename', name, '-']
        result = subprocess.run(command, input=content, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (qa / (label + '-ruff.json')).write_bytes(result.stdout)
        issues = json.loads(result.stdout)
        lines = content.decode().splitlines()
        signatures = collections.Counter((item['code'], item['message'], lines[item['location']['row'] - 1].strip()) for item in issues)
        fmt_command = [py, '-m', 'ruff', 'format', '--check', '--stdin-filename', name, '-']
        fmt = subprocess.run(fmt_command, input=content, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (qa / (label + '-format.log')).write_bytes(fmt.stdout)
        entries.append({'phase': phase, 'ruff_exit': result.returncode, 'diagnostics': len(issues), 'format_exit': fmt.returncode, 'source_sha256': hashlib.sha256(content).hexdigest(), 'signatures': signatures})
    same = entries[0]['signatures'] == entries[1]['signatures']
    summary[name] = {'runs': [{key: value for key, value in entry.items() if key != 'signatures'} for entry in entries], 'same_ruff_signatures': same}
    print(name, [(entry['ruff_exit'], entry['diagnostics'], entry['format_exit']) for entry in entries], 'same', same)
    assert same, name
    if name.endswith('test_console_trace_service.py'):
        assert current.startswith(baseline)
        snippet = current[len(baseline):].lstrip(b'\n')
        fmt = subprocess.run([py, '-m', 'ruff', 'format', '--stdin-filename', name, '-'], input=snippet, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert fmt.stdout == snippet
        summary[name]['added_test_formatted'] = True
base_service = subprocess.run(['git', 'show', base_ref + ':tldw_chatbook/Chat/console_trace_service.py'], check=True, stdout=subprocess.PIPE).stdout.decode()
current_service = Path('tldw_chatbook/Chat/console_trace_service.py').read_text()
unchanged = []
for name in ['_validate_completed_tool_turn', '_completed_tool_turn_parent', 'prepare_current_surface_delta', 'persist_request', '_validate_replacement_projection']:
    segments = []
    for source in [base_service, current_service]:
        node = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name)
        segments.append(ast.get_source_segment(source, node))
    assert segments[0] == segments[1], name
    unchanged.append(name)
summary['byte_unchanged_service_guards'] = unchanged
(qa / 'final-static-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Byte-unchanged guards:', unchanged)
