"""Provide immutable dependency excerpts for named Task8 review risks."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
HEAD = "40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77"
SCOPE = {
    "tldw_chatbook/Chat/console_chat_controller.py": (
        "begin_session_close",
        "finalize_session_close",
        "_chat_create_source_is_open",
        "prepare_agent_chat_create",
        "_chat_creation_source_live",
    ),
    "tldw_chatbook/Chat/console_chat_store.py": ("sessions",),
    "tldw_chatbook/Chat/console_chat_start.py": ("_source_live",),
}


def main():
    chunks = [
        f"# Immutable Task8 dependencies at {HEAD}\n\nNamed risks: committed Close atomicity; no lock reentry in source-open predicate; actual child/prepared liveness before Close; native preaccept source fence.\n"
    ]
    rows = []
    for path, names in SCOPE.items():
        data = subprocess.check_output(
            ["git", "-c", "gc.auto=0", "show", f"{HEAD}:{path}"], cwd=ROOT
        )
        source = data.decode()
        lines = source.splitlines(keepends=True)
        nodes = {}
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            ):
                assert node.name not in nodes
                nodes[node.name] = node
        assert set(nodes) == set(names)
        for name in names:
            node = nodes[name]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            snippet = "".join(lines[start - 1 : node.end_lineno])
            rows.append(
                {
                    "path": path,
                    "method": name,
                    "line": start,
                    "end": node.end_lineno,
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                    "snippet_sha256": hashlib.sha256(snippet.encode()).hexdigest(),
                }
            )
            chunks.append(f"\n## {path}:{start} — {name}\n\n```python\n{snippet}```\n")
    output = SDD / f"task-8-immutable-dependency-methods-{HEAD[:12]}.md"
    receipt = output.with_suffix(".json")
    assert not output.exists() and not receipt.exists()
    data = "".join(chunks).encode()
    output.write_bytes(data)
    row = {
        "frozen_head": HEAD,
        "path": str(output),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "snippets": rows,
    }
    receipt.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ("frozen_head", "path", "sha256", "bytes")}))


if __name__ == "__main__":
    main()
