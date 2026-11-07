"""Compare PR3166 changed-file Ruff findings with exact current-dev sources."""

import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = "b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4"
paths = [
    p
    for p in subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "--", "tldw_chatbook", "Tests"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if p.endswith(".py")
]
results = []


def findings(path, source):
    run = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format=json",
            "--stdin-filename",
            path,
            "-",
        ],
        input=source,
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=ROOT,
    )
    if run.returncode not in (0, 1):
        raise RuntimeError(run.stderr)
    lines = source.splitlines()
    return Counter(
        (row["code"], row["message"], lines[row["location"]["row"] - 1].strip())
        for row in json.loads(run.stdout)
    )


for path in paths:
    current = (ROOT / path).read_text(encoding="utf-8")
    ast.parse(current, filename=path)
    original = subprocess.run(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT, capture_output=True
    )
    before = (
        findings(path, original.stdout.decode("utf-8"))
        if original.returncode == 0
        else Counter()
    )
    after = findings(path, current)
    added = after - before
    results.append(
        {
            "path": path,
            "before": sum(before.values()),
            "after": sum(after.values()),
            "added": [
                {"code": k[0], "message": k[1], "line": k[2], "count": v}
                for k, v in added.items()
            ],
        }
    )
Path(__file__).with_name("lint-comparison.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "files_parsed": len(paths),
            "before": sum(x["before"] for x in results),
            "after": sum(x["after"] for x in results),
            "added": [x for x in results if x["added"]],
        }
    )
)
raise SystemExit(any(x["added"] for x in results))
