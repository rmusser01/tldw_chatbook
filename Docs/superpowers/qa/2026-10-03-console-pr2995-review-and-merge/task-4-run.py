import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path.cwd()
evidence = root / '.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'
name, *argv = sys.argv[1:]
env = os.environ.copy()
if '-m' in argv and 'pytest' in argv:
    profile = tempfile.mkdtemp(prefix=f'task-4-{name}-profile-', dir=evidence)
    env['TLDW_TEST_CONFIG_ROOT'] = profile
    argv.append('--basetemp=' + profile + '/pytest')
    argv.append('--junitxml=' + str(evidence / f'task-4-{name}.xml'))
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
paths = [
    'tldw_chatbook/Chat/console_chat_start.py',
    'tldw_chatbook/Chat/console_chat_controller.py',
    'tldw_chatbook/Chat/message_metadata.py',
    'tldw_chatbook/DB/automatic_work.py',
    'Tests/Chat/test_console_chat_start.py',
    'Tests/DB/test_automatic_chat_starts.py',
    'Tests/Agents/test_automatic_child_scope.py',
    'Tests/Chat/test_console_durable_commit_offload.py',
    'Tests/Chat/test_console_durable_commit_diagnostics.py',
    'Tests/Chat/test_console_close_during_durable_postcommit.py',

]
receipt = {'name': name, 'revision': revision, 'cwd': str(root), 'argv': argv,
           'environment': {'TLDW_TEST_CONFIG_ROOT': env.get('TLDW_TEST_CONFIG_ROOT')},
           'sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}}
with (evidence / f'task-4-{name}.log').open('w') as log:
    result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
receipt['exit'] = result.returncode
(evidence / f'task-4-{name}-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'sha256'}, indent=2))
print((evidence / f'task-4-{name}.log').read_text()[-12000:])
sys.exit(result.returncode)
