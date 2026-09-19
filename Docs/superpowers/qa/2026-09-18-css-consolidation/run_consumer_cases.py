"""Run an explicit targeted case list serially, each in a fresh private profile."""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = Path(sys.argv[1]).resolve()
OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=False)
selections = Path(sys.argv[2]).read_text().splitlines()
selections = [
    line.strip() for line in selections if line.strip() and not line.startswith("#")
]


def environment(directory, node=None):
    profile = directory / "profile"
    profile.mkdir(mode=0o700, parents=True)
    result = dict(os.environ)
    for key in (
        "TLDW_TEST_CONFIG_ROOT_OWNER",
        "PYTEST_ADDOPTS",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "TLDW_TEST_PRIVATE_PROFILE_NODE",
    ):
        result.pop(key, None)
    result.update(
        HOME=str(profile / "home"),
        USERPROFILE=str(profile / "home"),
        XDG_CONFIG_HOME=str(profile / "config"),
        XDG_DATA_HOME=str(profile / "data"),
        TLDW_CONFIG_PATH=str(profile / "config" / "config.toml"),
        TLDW_TEST_CONFIG_ROOT=str(profile),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="migration_parity",
        PYTHONPATH=f"{HERE}:{REPO}",
        TLDW_CSS_PARITY_REPORT_DIR=str(directory / "parity"),
    )
    if node:
        result.update(
            TLDW_TEST_PRIVATE_PROFILE_NODE=node, TLDW_CSS_PARITY_ISOLATED_NODE="1"
        )
    return result


base = [
    sys.executable,
    "-m",
    "pytest",
    "-p",
    "pytest_asyncio.plugin",
    "-p",
    "pytest_timeout",
    "-q",
    "-rP",
]
collection = subprocess.run(
    base + ["--collect-only", *selections],
    cwd=REPO,
    check=False,
    env=environment(OUTPUT / "collection"),
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
(OUTPUT / "collection.log").write_text(collection.stdout)
if collection.returncode:
    raise SystemExit(collection.returncode)
nodes = [
    line
    for line in collection.stdout.splitlines()
    if line.startswith("Tests/") and "::" in line
]
assert nodes and len(nodes) == len(set(nodes)), (
    "collection must select unique concrete cases"
)
results = []
for index, node in enumerate(nodes, 1):
    directory = OUTPUT / f"{index:03d}"
    env = environment(directory, node)
    with (directory / "pytest.log").open("w") as log:
        run = subprocess.run(
            base + ["--timeout=180", node],
            cwd=REPO,
            check=False,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=200,
        )
    reports = [
        json.loads(path.read_text()) for path in (directory / "parity").glob("*.json")
    ]
    assert len(reports) == 1 and reports[0]["cases"] == [node], (
        "exact original case must execute"
    )
    calls = [row for row in reports[0]["outcomes"] if row["phase"] == "call"]
    qualified = (
        run.returncode == 0
        and len(calls) == 1
        and calls[0]["outcome"] == "passed"
        and calls[0]["wasxfail"] is None
    )
    results.append(
        {
            "node": node,
            "returncode": run.returncode,
            "qualified": qualified,
            "outcomes": reports[0]["outcomes"],
            "directory": str(directory),
            "classes": sorted(reports[0]["classes"]),
            "mismatches": len(reports[0]["mismatches"]),
        }
    )
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2))
    print(
        f"{index}/{len(nodes)} {'PASS' if qualified else 'FAIL'} {node}",
        flush=True,
    )
raise SystemExit(1 if not all(row["qualified"] for row in results) else 0)
