"""Check new lint debt with the pinned tool and verify frozen harness semantics."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUFF = ROOT / ".tmp-tools/ruff-0.16.6/bin/ruff.exe"
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
BASE = "cc6006fcf59328353004206bba8724c399216749"
version = subprocess.check_output([str(RUFF), "--version"], text=True).strip()
assert version == "ruff 0.16.6"
files = subprocess.check_output(
    GIT + ["diff", "--name-only", BASE, "--", "tldw_chatbook", "Tests"],
    cwd=ROOT,
    text=True,
).splitlines()
result = {"ruff": version, "base": BASE, "files": {}, "executed_harness": {}}
for file in files:
    if not file.endswith(".py"):
        continue
    current = (ROOT / file).read_bytes()
    old = subprocess.run(
        GIT + ["show", f"{BASE}:{file}"], cwd=ROOT, capture_output=True
    )
    counts = []
    details = []
    for data in (old.stdout if old.returncode == 0 else b"", current):
        lint = subprocess.run(
            [
                str(RUFF),
                "check",
                "--output-format",
                "json",
                "--stdin-filename",
                file,
                "-",
            ],
            cwd=ROOT,
            input=data,
            capture_output=True,
        )
        assert lint.returncode in (0, 1), lint.stderr.decode()
        findings = json.loads(lint.stdout)
        counts.append(Counter((item["code"], item["message"]) for item in findings))
        details.append(findings)
    introduced = counts[1] - counts[0]
    assert not introduced, (file, introduced)
    result["files"][file] = {
        "introduced": 0,
        "existing": sum(counts[1].values()),
        "codes": sorted({item["code"] for item in details[1]}),
    }
frozen = json.loads((OUT / "executed-harness.json").read_text())
for file, metadata in frozen.items():
    data = (OUT / metadata["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
    assert ast.dump(ast.parse(data)) == ast.dump(
        ast.parse((OUT / file).read_bytes())
    ), file
    result["executed_harness"][file] = {
        "exact_snapshot_sha256": metadata["sha256"],
        "current_ast_equal": True,
    }
guards = json.loads((OUT / "safety-baseline.json").read_text())
changes = [
    path
    for path, digest in guards["sha256"].items()
    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest
]
assert sorted(changes) == [
    "tldw_chatbook/Chat/console_context_repository.py",
    "tldw_chatbook/Chat/console_preparation_reads.py",
], changes
result["safety_modules"] = {
    "unchanged": len(guards["sha256"]) - len(changes),
    "intentional_close_fixes": changes,
}
(OUT / "static-checks.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "files_checked": len(result["files"]),
            "new_findings": 0,
            "safety": result["safety_modules"],
            "frozen_harness_ast_equal": True,
        }
    )
)
