"""Freeze a scoped Task7 review package after the source worker finishes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
BASE = "223c4db6d15f0ca9e9e3da844cc81cd1f2a68aa1"
MODEL_BASE = "6323ff208b523496dd29341fccb49e023a2ca1b5"
MODEL_FIX = "9ed370f5668d82d9e6a0108835aa87136a57ce74"


def git(*args):
    return subprocess.check_output(["git", "-c", "gc.auto=0", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_head")
    args = parser.parse_args()
    frozen = git("rev-parse", args.frozen_head).decode().strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, frozen], cwd=ROOT, check=True
    )
    changed = git("diff", "--name-only", BASE, frozen).decode().splitlines()
    excluded = {
        "Docs/superpowers/plans/2026-10-03-console-pr2995-review-and-merge.md",
        "backlog/tasks/task-34215.2 - Integrate-Console-chat-starts-with-current-dev-before-publication.md",
    }
    selected = [path for path in changed if path not in excluded]
    assert selected
    assert not any(path.startswith("Docs/superpowers/qa/") for path in selected)
    chunks = [
        f"Task7 frozen source {BASE} -> {frozen}\n".encode(),
        b"Commit list\n",
        git("log", "--oneline", "--reverse", f"{BASE}..{frozen}"),
        b"Task7 stat\n",
        git("diff", "--stat", BASE, frozen, "--", *selected),
        git("diff", "--binary", BASE, frozen, "--", *selected),
        b"\nPrior model-config two-test correction only\n",
        git(
            "diff",
            MODEL_BASE,
            MODEL_FIX,
            "--",
            "Tests/UI/test_console_native_chat_flow.py",
            "Tests/Chat/test_console_provider_gateway.py",
        ),
    ]
    data = b"\n".join(chunks)
    output = SDD / f"task-7-review-package-{frozen[:12]}.diff"
    receipt = SDD / f"task-7-review-package-{frozen[:12]}.json"
    assert not output.exists() and not receipt.exists()
    output.write_bytes(data)
    row = {
        "base": BASE,
        "frozen_head": frozen,
        "scope": "Task7 behavior/extraction and prior two model-config test corrections only; earlier broad/schema/scanner reviews carry by exact mapping.",
        "selected_paths": selected,
        "source_blobs": {
            path: hashlib.sha256(git("show", f"{frozen}:{path}")).hexdigest()
            for path in selected
        },
        "package": str(output),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "earlier_test_fix": {"base": MODEL_BASE, "head": MODEL_FIX},
        "reviewer_must_not_replay_tests": True,
    }
    receipt.write_text(json.dumps(row, indent=2) + "\n")
    print(
        json.dumps(
            {key: row[key] for key in ("frozen_head", "package", "sha256", "bytes")}
        )
    )


if __name__ == "__main__":
    main()
