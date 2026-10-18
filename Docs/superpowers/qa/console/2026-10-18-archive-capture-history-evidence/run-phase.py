import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

phase = sys.argv[1]
command = sys.argv[2:]
qa = Path('Docs/superpowers/qa/console/2026-10-18-archive-capture-history-evidence')
qa.mkdir(parents=True, exist_ok=True)
source_hashes = {
    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for name in [
        'tldw_chatbook/Chat/console_chat_controller.py',
        'tldw_chatbook/Chat/console_runtime.py',
        'tldw_chatbook/Chat/console_agent_bridge.py',
        'Tests/Chat/test_console_archive_recovery_followup.py',
        'tldw_chatbook/Chat/console_trace_service.py',
        'Tests/Chat/test_console_trace_service.py',
    ]
}
env = {'PYTHONPATH': '.', 'PYTHONDONTWRITEBYTECODE': '1'}
result = subprocess.run(command, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(qa / (phase + '.log')).write_bytes(result.stdout)
(qa / (phase + '-command.json')).write_text(json.dumps({
    'cwd': str(Path.cwd()), 'env': env, 'argv': command,
    'exit': result.returncode, 'source_sha256': source_hashes,
}, indent=2) + '\n')
print(result.stdout.decode()[:6500] if result.returncode else result.stdout.decode()[-4000:])
print('PHASE', phase, 'EXIT', result.returncode)
raise SystemExit(result.returncode)
