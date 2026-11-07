import json
from pathlib import Path
import subprocess
import sys
import time

artifacts = Path('/private/tmp/mixins-97a-archive-fix')
command = [
    sys.executable, '-m', 'pytest',
    'Tests/UI/test_console_archive_recovery_cards.py',
    'Tests/UI/test_console_partial_resize.py',
    'Tests/UI/test_console_archive_recovery_flow.py::test_pending_source_remount_expiry_and_detach_fail_closed',
    'Tests/UI/test_console_archive_recovery_flow.py::test_restore_as_requires_fresh_final_name_confirmation',
    'Tests/UI/test_console_archive_recovery_flow.py::test_compact_confirmation_focus_shows_exact_target',
    '-q', '--basetemp=' + str(artifacts / 'pytest-frozen-native'),
    '--junitxml=' + str(artifacts / 'frozen-native.xml'),
]
started = time.monotonic()
with (artifacts / 'frozen-native.log').open('w') as output:
    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
receipt = {'argv': command, 'exit_code': result.returncode, 'duration_seconds': time.monotonic() - started}
(artifacts / 'frozen-native-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
