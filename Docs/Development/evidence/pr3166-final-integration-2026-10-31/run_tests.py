"""PR3166 final integration focused tests in native private Windows roots."""

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
private_child_node = None
if args and args[0] == "--private-node":
    _, private_child_node, *args = args
    assert private_child_node in args and "::" in private_child_node
run = Path.home() / (".tmp-pr3166-final-" + uuid.uuid4().hex[:12])
for path in (run, run / "config", run / "temp"):
    native_os.mkdir(path, 0o700)
env = dict(
    os.environ,
    TLDW_TEST_CONFIG_ROOT=str(run / "config"),
    TEMP=str(run / "temp"),
    TMP=str(run / "temp"),
    PYTHONUTF8="1",
)
if private_child_node is not None:
    # The supported decorator child entry, already in its own native-private
    # process/profile. Avoid a second nested process/profile; keep native fixture ownership
    # active in the exact test process.
    env["TLDW_TEST_PRIVATE_PROFILE_NODE"] = private_child_node
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    args += ["-p", "pytest_asyncio.plugin", "-p", "pytest_timeout"]
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
Path(__file__).with_name(label + ".patch").write_bytes(source_diff)
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
    "private_child_node": private_child_node,
    "command": command,
    "fixture_root": str(run),
    "exit_code": result.returncode,
    "seconds": time.monotonic() - started,
}
Path(__file__).with_name(label + ".json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
print("\n".join(line[:1000] for line in log.read_text(encoding="utf-8").splitlines()[-30:]), flush=True)
raise SystemExit(result.returncode)
