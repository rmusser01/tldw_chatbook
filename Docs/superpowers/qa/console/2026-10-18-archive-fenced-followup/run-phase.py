"""Record exact offline command outcomes without importing production code."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

qa = Path(__file__).resolve().parent
phase, *command = sys.argv[1:]
paths = [
    "tldw_chatbook/Chat/console_chat_controller.py",
    "tldw_chatbook/Agents/agent_runtime.py",
    "Tests/Chat/test_console_archive_recovery_followup.py",
]
env = {"PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"}
result = subprocess.run(
    command,
    env={**os.environ, **env},
    stdout=subprocess.PIPE,
    check=False,
    stderr=subprocess.STDOUT,
)
log = qa / f"{phase}.log"
log.write_bytes(result.stdout)
(qa / f"{phase}-command.json").write_text(
    json.dumps(
        {
            "cwd": str(Path.cwd()),
            "argv": command,
            "env": env,
            "interpreter": sys.version,
            "exit": result.returncode,
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in paths
            },
            "log_sha256": hashlib.sha256(result.stdout).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
print(result.stdout.decode()[-6500:])
print("PHASE", phase, "EXIT", result.returncode)
raise SystemExit(result.returncode)
