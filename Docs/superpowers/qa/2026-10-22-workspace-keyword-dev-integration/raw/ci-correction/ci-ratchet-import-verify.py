from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()
evidence = root / ".superpowers/sdd/2026-10-22-keyword-dev-integration"
rel = "tldw_chatbook/Chat/answer_citations.py"
path = root / rel
git = ["git", "-c", "safe.directory=C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui"]
python = root / ".venv/Scripts/python.exe"
ruff = Path("C:/Python312/Scripts/ruff.exe")

def run(argv: list[str]) -> dict:
    p = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    return {"command": argv, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

base_command = [*git, "show", "HEAD:" + rel]
base_result = run(base_command)
assert base_result["exit_code"] == 0, base_result
before = base_result["stdout"]
after = path.read_text(encoding="utf-8")
old = "from dataclasses import dataclass\nfrom tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole\nfrom typing import Any, Mapping\n\nfrom tldw_chatbook.Chat.citation_evidence_models import ("
new = "from dataclasses import dataclass\nfrom typing import Any, Mapping\n\nfrom tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole\nfrom tldw_chatbook.Chat.citation_evidence_models import ("
assert before.count(old) == 1
assert after == before.replace(old, new), "unexpected text changes"
old_tree = ast.parse(before, filename=rel)
new_tree = ast.parse(after, filename=rel)
is_import = lambda node: isinstance(node, (ast.Import, ast.ImportFrom))
old_imports = sorted(ast.dump(node, include_attributes=False) for node in old_tree.body if is_import(node))
new_imports = sorted(ast.dump(node, include_attributes=False) for node in new_tree.body if is_import(node))
old_body = ast.dump(ast.Module(body=[node for node in old_tree.body if not is_import(node)], type_ignores=[]), include_attributes=False)
new_body = ast.dump(ast.Module(body=[node for node in new_tree.body if not is_import(node)], type_ignores=[]), include_attributes=False)
compile(after, rel, "exec")
checks = [
    run([str(ruff), "check", rel]),
    run([str(ruff), "format", "--check", rel]),
    run([*git, "diff", "--check"]),
    run([*git, "diff", "--", rel]),
]
receipt = {
    "base_command": base_command,
    "base_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
    "current_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
    "exact_text_only_reviewed_import_move": True,
    "top_level_import_multiset_equal": old_imports == new_imports,
    "non_import_module_ast_equal": old_body == new_body,
    "compile": "passed",
    "checks": checks,
}
assert receipt["top_level_import_multiset_equal"]
assert receipt["non_import_module_ast_equal"]
assert all(c["exit_code"] == 0 for c in checks[:3]), checks[:3]
(evidence / "ci-ratchet-import-verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print("base_sha256=" + receipt["base_sha256"])
print("current_sha256=" + receipt["current_sha256"])
print("exact_text_only_reviewed_import_move=True")
print("top_level_import_multiset_equal=True")
print("non_import_module_ast_equal=True")
print("compile=passed")
for c in checks:
    print("command=" + " ".join(c["command"]))
    print("exit_code=" + str(c["exit_code"]))
    print("stdout=" + c["stdout"].rstrip())
    print("stderr=" + c["stderr"].rstrip())

