"""Check Ruff diagnostics and formatting on authored Python lines."""

import json
import re
import subprocess
from pathlib import Path

BASE = "e5ac111967bd7310e6e97dec043a559d07e97d30"
root = Path.cwd()
paths = subprocess.check_output(
    ["git", "diff", "--name-only", BASE], text=True
).splitlines()
paths += subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard"], text=True
).splitlines()
paths = sorted(
    {
        p
        for p in paths
        if p.endswith(".py") and p.startswith(("tldw_chatbook/", "Tests/"))
    }
)
ranges = {}
new = []
for path in paths:
    exists = (
        subprocess.run(
            ["git", "cat-file", "-e", BASE + ":" + path],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not exists:
        new.append(path)
        ranges[path] = [(1, len(Path(path).read_text().splitlines()) + 1)]
    else:
        diff = subprocess.check_output(
            ["git", "diff", "--unified=0", BASE, "--", path], text=True
        )
        ranges[path] = [
            (int(start), int(start) + int(length or 1))
            for start, length in re.findall(
                r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE
            )
            if length != "0"
        ]
check = subprocess.run(
    [".venv/bin/ruff", "check", "--no-cache", "--output-format=json", *paths],
    capture_output=True,
    text=True,
    check=False,
)
diagnostics = json.loads(check.stdout)
authored = []
for diagnostic in diagnostics:
    path = str(Path(diagnostic["filename"]).relative_to(root))
    line = diagnostic["location"]["row"]
    if any(start <= line < end for start, end in ranges[path]):
        authored.append(
            {
                "path": path,
                "code": diagnostic["code"],
                "line": line,
                "message": diagnostic["message"],
            }
        )
format_failures = []
for path, spans in ranges.items():
    for start, end in spans:
        result = subprocess.run(
            [
                ".venv/bin/ruff",
                "format",
                "--no-cache",
                "--check",
                f"--range={start}-{end}",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            format_failures.append({"path": path, "range": [start, end]})
report = {
    "base": BASE,
    "checked_files": len(paths),
    "new_files": new,
    "all_diagnostics": len(diagnostics),
    "diagnostics_on_authored_lines": authored,
    "authored_range_format_failures": format_failures,
}
Path("/private/tmp/hook-review-final-static.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
raise SystemExit(bool(authored or format_failures))
