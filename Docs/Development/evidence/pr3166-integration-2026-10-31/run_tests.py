"""PR3166 current-dev focused tests in native private Windows roots."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tldw_chatbook.Utils.platform_files import os as native_os  # noqa: E402

label, *args = sys.argv[1:]
run = Path.home() / (".tmp-pr3166-integration-" + uuid.uuid4().hex[:12])
for path in (run, run / "config", run / "temp"):
    native_os.mkdir(path, 0o700)
env = dict(
    os.environ,
    TLDW_TEST_CONFIG_ROOT=str(run / "config"),
    TEMP=str(run / "temp"),
    TMP=str(run / "temp"),
    PYTHONUTF8="1",
)
command = [
    sys.executable,
    "-X",
    "utf8",
    "-m",
    "Tests.windows_private_fixture_runner",
    *args,
]
source_diff = subprocess.check_output(
    ["git", "diff", "HEAD", "--", "tldw_chatbook", "Tests"], cwd=ROOT
)
started = time.monotonic()
log = Path(__file__).with_name(label + ".log")
with log.open("w", encoding="utf-8") as output:
    result = subprocess.run(
        command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
    )
receipt = {
    "head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
    "command": command,
    "fixture_root": str(run),
    "exit_code": result.returncode,
    "seconds": time.monotonic() - started,
}
Path(__file__).with_name(label + ".json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
print("\n".join(log.read_text(encoding="utf-8").splitlines()[-60:]), flush=True)
raise SystemExit(result.returncode)
