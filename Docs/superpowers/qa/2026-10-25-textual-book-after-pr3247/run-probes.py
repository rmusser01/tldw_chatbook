"""Reproduce the bounded audit observations in fresh private profiles."""

from importlib import import_module
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

package = Path(__file__).resolve().parent
root = package.parents[3]
assert (root / "tldw_chatbook").is_dir(), root
sys.path.insert(0, str(root))
platform_files = import_module("tldw_chatbook.Utils.platform_files")

label = sys.argv[1]
assert label.replace("-", "").replace("_", "").isalnum()
run = root / ".test-tmp/pr3247-audit-20261025" / label
platform_files.os.mkdir(run, 0o700)
git = ["git", "-c", "safe.directory=" + root.as_posix()]
head = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=root, text=True).strip()
paths = ['tldw_chatbook/UI/Screens/study_screen.py', 'tldw_chatbook/UI/Study_Window.py', 'tldw_chatbook/UI/Study_Modules/quizzes_handler.py', 'tldw_chatbook/Study_Interop/quiz_scope_service.py', 'tldw_chatbook/Study_Interop/server_quiz_service.py', 'tldw_chatbook/Study_Interop/local_quiz_service.py', 'tldw_chatbook/UI/Screens/watchlists_collections_screen.py', 'tldw_chatbook/UI/Watchlists_Modules/article_list.py', 'tldw_chatbook/UI/Watchlists_Modules/sources_pane.py', 'tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py', 'tldw_chatbook/Subscriptions/watchlist_scope_service.py', 'tldw_chatbook/Subscriptions/local_watchlists_service.py', 'tldw_chatbook/Subscriptions/db_offload.py', 'tldw_chatbook/DB/Subscriptions_DB.py']

pins = {
    p: {
        "blob": subprocess.check_output(
            git + ["rev-parse", "HEAD:" + p], cwd=root, text=True
        ).strip(),
        "working_sha256": hashlib.sha256((root / p).read_bytes()).hexdigest(),
    }
    for p in paths
}
(run / "source-pins.json").write_text(
    json.dumps({"head": head, "paths": pins}, indent=2), encoding="utf-8"
)


def execute(module):
    out = run / module
    for p in [out, out / "tmp", out / "bootstrap"]:
        platform_files.os.mkdir(p, 0o700)
    for name in ["conftest.py", "test_" + module + ".py"]:
        shutil.copyfile(package / name, out / name)
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        TEMP=str(out / "tmp"),
        TMP=str(out / "tmp"),
        TLDW_TEST_CONFIG_ROOT=str(out / "bootstrap"),
    )
    args = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "pytest",
        str(out / ("test_" + module + ".py")),
        "-p",
        "pytest_asyncio.plugin",
        "-p",
        "pytest_timeout",
        "-q",
        "--tb=short",
        "--timeout=60",
        "-o",
        "cache_dir=" + str(out / "cache"),
        "--basetemp=" + str(out / "pytest"),
        "--junitxml=" + str(out / "result.xml"),
    ]
    (out / "command.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    result = subprocess.run(
        args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8"
    )
    (out / "run.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    suites = (
        list(ET.parse(out / "result.xml").iter("testsuite"))
        if (out / "result.xml").exists()
        else []
    )
    receipt = {
        "module": module,
        "exit_code": result.returncode,
        "tests": sum(int(s.get("tests", 0)) for s in suites),
        "failures": sum(int(s.get("failures", 0)) for s in suites),
        "errors": sum(int(s.get("errors", 0)) for s in suites),
        "seconds": sum(float(s.get("time", 0)) for s in suites),
    }
    print(json.dumps(receipt), flush=True)
    if result.returncode:
        print(result.stdout[-5000:], flush=True)
    return receipt


with ThreadPoolExecutor(max_workers=3) as pool:
    receipts = list(pool.map(execute, ["study", "watchlists", "sources"]))
(run / "verification.json").write_text(
    json.dumps({"head": head, "results": receipts}, indent=2), encoding="utf-8"
)
assert all(
    x["exit_code"] == 0 and x["failures"] == x["errors"] == 0 for x in receipts
), receipts
assert {x["module"]: x["tests"] for x in receipts} == {
    "study": 6,
    "watchlists": 3,
    "sources": 4,
}
print(
    "Thirteen observations/controls passed; these reproduce unfixed audit behavior, not repair regressions."
)
