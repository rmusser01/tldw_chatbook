"""Freeze only the actual Task8 collision union and corrective delta."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
BASE = "edaeececfb3733885bb5fec3ea7b5823266dff23"
DEV = "5e0341d1ec701865e019eb2fd8a5e2028ab2d474"
REPLAY = "bc184be7828a71b178256580a06980d4ee9bf5eb"
CONTROLLER = "tldw_chatbook/Chat/console_chat_controller.py"
COLLISIONS = ("__init__", "request_chat_create_confirm", "execute_agent_chat_create")
METADATA = {
    "Docs/superpowers/plans/2026-10-03-console-pr2995-review-and-merge.md",
    "backlog/tasks/task-34215.2 - Integrate-Console-chat-starts-with-current-dev-before-publication.md",
}


def git(*args):
    return subprocess.check_output(["git", "-c", "gc.auto=0", *args], cwd=ROOT)


def controller_methods(revision):
    data = git("show", f"{revision}:{CONTROLLER}")
    text = data.decode()
    owner = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == "ConsoleChatController"
    )
    lines = text.splitlines(keepends=True)
    result = {}
    for node in owner.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in COLLISIONS
        ):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result[node.name] = (start, "".join(lines[start - 1 : node.end_lineno]))
    assert set(result) == set(COLLISIONS)
    return data, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_head")
    args = parser.parse_args()
    head = git("rev-parse", args.frozen_head).decode().strip()
    for ancestor in (DEV, REPLAY):
        subprocess.run(
            ["git", "-c", "gc.auto=0", "merge-base", "--is-ancestor", ancestor, head],
            cwd=ROOT,
            check=True,
        )
    paths = git("diff", "--name-only", REPLAY, head).decode().splitlines()
    paths = [p for p in paths if p not in METADATA]
    assert paths and not any(p.startswith("Docs/superpowers/qa/") for p in paths)
    chunks = [
        f"Task8 scoped integration review\nFeature checkpoint {BASE}\nIncoming {DEV}\nReplayed union {REPLAY}\nFinal source {head}\n".encode(),
        b"Part1: actual three-method collision inputs and replayed union (immutable source; exact line labels).\n",
    ]
    source_map = {}
    for label, revision in (
        ("qualified feature", BASE),
        ("incoming dev", DEV),
        ("replayed union", REPLAY),
    ):
        data, methods = controller_methods(revision)
        source_map[label] = {
            "revision": revision,
            "controller_sha256": hashlib.sha256(data).hexdigest(),
            "methods": {},
        }
        for name in COLLISIONS:
            line, snippet = methods[name]
            source_map[label]["methods"][name] = {
                "line": line,
                "sha256": hashlib.sha256(snippet.encode()).hexdigest(),
            }
            chunks.append(
                f"\n{label}: {revision}:{CONTROLLER}:{line} — {name}\n".encode()
            )
            chunks.append(snippet.encode())
    chunks.extend(
        [
            b"\nPart2: complete corrective source/test/derived delta from replayed union to final frozen source.\n",
            git("log", "--oneline", "--reverse", f"{REPLAY}..{head}"),
            git("diff", "--stat", REPLAY, head, "--", *paths),
            git("diff", "-U30", "--binary", REPLAY, head, "--", *paths),
        ]
    )
    package = b"\n".join(chunks)
    output = SDD / f"task-8-review-package-{head[:12]}.diff"
    receipt = output.with_suffix(".json")
    assert not output.exists() and not receipt.exists()
    output.write_bytes(package)
    payload = {
        "feature_checkpoint": BASE,
        "incoming": DEV,
        "replayed": REPLAY,
        "frozen_head": head,
        "scope": "Actual three production collision inputs/replayed union and every subsequent corrective source/test/derived delta. Exact incoming nonoverlaps and prior reviewed feature owners carry by separate full preservation maps. Not a second whole-feature/schema/scanner review.",
        "selected_corrective_paths": paths,
        "collision_inputs": source_map,
        "final_source_blobs": {
            p: hashlib.sha256(git("show", f"{head}:{p}")).hexdigest() for p in paths
        },
        "package": str(output),
        "sha256": hashlib.sha256(package).hexdigest(),
        "bytes": len(package),
        "reviewer_must_not_replay_tests": True,
    }
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {k: payload[k] for k in ("frozen_head", "package", "sha256", "bytes")}
        )
    )


if __name__ == "__main__":
    main()
