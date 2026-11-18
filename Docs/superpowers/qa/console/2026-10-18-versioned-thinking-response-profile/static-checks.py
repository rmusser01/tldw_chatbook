import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess

base = '38cf68612508d23e9ae70a5deecf76a913fff3e0'
out = Path('/private/tmp/task4-static-final')
out.mkdir(exist_ok=True)
files = [
    'tldw_chatbook/Chat/console_semantic_revision.py',
    'tldw_chatbook/Chat/console_trace_service.py',
    'Tests/Chat/test_console_trace_settlement.py',
    'Tests/Chat/test_console_trace_thinking_settlement.py',
    'Tests/Chat/test_console_archive_recovery_followup.py',
]
ruff = ['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python', '-m', 'ruff']
summary = {}
for name in files:
    baseline = out / 'baseline' / name
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_bytes(subprocess.check_output(['git', 'show', base + ':' + name]))
    results = {}
    for phase, path in [('baseline', baseline), ('current', Path(name))]:
        lint = subprocess.run(ruff + ['check', '--no-cache', '--output-format=json', str(path)], capture_output=True)
        (out / (path.name + '.' + phase + '.ruff.json')).write_bytes(lint.stdout)
        fmt = subprocess.run(ruff + ['format', '--no-cache', '--check', str(path)], capture_output=True)
        (out / (path.name + '.' + phase + '.format.log')).write_bytes(fmt.stdout + fmt.stderr)
        diagnostics = json.loads(lint.stdout)
        results[phase] = {'lint_exit': lint.returncode, 'diagnostics': len(diagnostics), 'format_exit': fmt.returncode, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        signatures = collections.Counter((x['code'], x['message']) for x in diagnostics)
        if phase == 'baseline':
            prior = signatures
        else:
            assert signatures == prior, (name, signatures - prior)
            assert fmt.returncode == 0, name
    summary[name] = results
    print(name, results)
for name in ['console_trace_settlement.py', 'console_provider_gateway.py', 'console_trace_runtime.py', 'console_chat_controller.py', 'console_trace_final_values.py', 'console_trace_repository.py']:
    path = Path('tldw_chatbook/Chat') / name
    baseline = subprocess.check_output(['git', 'show', base + ':' + str(path)])
    assert baseline == path.read_bytes(), path
    summary[str(path)] = {'unchanged': True, 'sha256': hashlib.sha256(baseline).hexdigest()}
path = 'tldw_chatbook/Chat/console_trace_service.py'
old = ast.parse(subprocess.check_output(['git', 'show', base + ':' + path]))
current = ast.parse(Path(path).read_bytes())
summary['guards'] = {}
for name in ['_validate_completed_tool_turn', '_completed_tool_turn_parent', 'prepare_current_surface_delta', 'persist_request', '_validate_replacement_projection']:
    before = next(n for n in ast.walk(old) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    after = next(n for n in ast.walk(current) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    assert ast.dump(before) == ast.dump(after), name
    summary['guards'][name] = hashlib.sha256(ast.dump(after).encode()).hexdigest()
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
whitespace = subprocess.run(['git', 'diff', '--check', '--', *files], capture_output=True)
(out / 'whitespace.log').write_bytes(whitespace.stdout + whitespace.stderr)
assert whitespace.returncode == 0
print('five strict guards unchanged; six boundary files unchanged; no additional Ruff signatures; all five files formatted; whitespace exit0')
