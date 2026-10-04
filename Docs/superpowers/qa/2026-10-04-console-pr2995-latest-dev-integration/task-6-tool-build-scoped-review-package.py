"""Build an immutable, task-scoped integration review without repeating history."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
REVIEWED_BASE = "b265d5bd2c2f5a28e4f56c2115f6247988d29ac2"
FIRST_REBASE = "52a3805727ed496723c8c2b18fb1c79e34cfbee4"
REPAIR_HEAD = "4a769ebb5353a68dc69c42d9d16042e2afa58833"
CHECKPOINT = "dc4df3dc80f767de7c94a833b0402dd880af235c"
PINNED_DEV = "ca2992cb10b24307fbae050643472ffb0a4388e7"
LATEST_REBASE = "9ce98df2e7cda8b444b3bed90f27c1d382677871"
READER_REPAIRS = [
    "Tests/UI/test_library_conversation_reader.py",
    "Tests/UI/test_library_conversation_reader_freshness.py",
    "tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py",
    "tldw_chatbook/UI/Library_Modules/library_conversations_state.py",
    "tldw_chatbook/UI/Library_Modules/library_skills_controller.py",
]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    final_head, final_dev, surface_path = sys.argv[1:]
    surface = json.loads(Path(surface_path).read_text())
    assert isinstance(surface, list) and all(isinstance(p, str) for p in surface)
    assert git("rev-parse", final_head).strip() == final_head
    assert git("rev-parse", final_dev).strip() == final_dev
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", final_dev, final_head],
        cwd=ROOT,
        check=True,
    )
    output = SDD / ("task-6-integration-review-" + final_head[:12] + ".diff")
    assert not output.exists()
    parts = [
        "# Immutable Task 6 integration review package\n",
        "Reviewed BASE: " + REVIEWED_BASE + "\n",
        "First rebase: " + FIRST_REBASE + "\n",
        "Corrective source: " + REPAIR_HEAD + "\n",
        "Prelatest metadata checkpoint: " + CHECKPOINT + "\n",
        "Final reviewed source: " + final_head + "\n",
        "Final upstream pin: " + final_dev + "\n",
        "\nThis is the complete functional review surface of Task 6. "
        "Formatting-only owned paths and retained upstream/QA paths are "
        "accounted for by the linked exact blob/strict AST manifests. "
        "Earlier Task 1-5 and whole-feature reviews remain bound to their "
        "original source hashes. They are not reviewed again here.\n",
        "\n## Initial canonical integration deltas\n",
        "These four canonical AST diffs retain literals, annotations and "
        "decorators; their display line numbers are canonical coordinates, "
        "not final source line numbers. Their merge proofs and final-source "
        "mappings are separate receipts.\n",
    ]
    canonical = sorted(SDD.glob("task-6-canonical-*.diff"))
    assert len(canonical) == 4
    for path in canonical:
        parts.extend(["\n### " + path.name + "\n", path.read_text()])
    parts.extend(
        [
            "\n## Corrective source and its tests\n",
            git("log", "--format=%H %s", FIRST_REBASE + ".." + REPAIR_HEAD),
            git("diff", "--stat", FIRST_REBASE, REPAIR_HEAD),
            git("diff", "--find-renames", "-U45", FIRST_REBASE, REPAIR_HEAD),
            "\n## Later upstream concrete integration seams\n",
            git("log", "--format=%H %s", PINNED_DEV + ".." + final_dev),
            "The following paths are the upstream seam surface selected from "
            "the actual dev delta; remaining upstream paths must retain exact "
            "upstream blobs in the preservation proof.\n",
            git("diff", "--stat", PINNED_DEV, final_dev, "--", *surface),
            git("diff", "-U35", PINNED_DEV, final_dev, "--", *surface),
            "\n## Corrective change after latest-dev qualification\n",
            git("log", "--format=%H %s", LATEST_REBASE + ".." + final_head),
            git(
                "diff",
                "--stat",
                LATEST_REBASE,
                final_head,
                "--",
                "Tests/UI/test_console_store_continuity.py",
            ),
            git(
                "diff",
                "-U45",
                LATEST_REBASE,
                final_head,
                "--",
                "Tests/UI/test_console_store_continuity.py",
            ),
            "\n## Reader fixture, format and comment-only repairs over final dev\n",
            git("diff", "--stat", final_dev, final_head, "--", *READER_REPAIRS),
            git("diff", "-U35", final_dev, final_head, "--", *READER_REPAIRS),
        ]
    )
    output.write_text("".join(parts))
    manifest = {
        "reviewed_base": REVIEWED_BASE,
        "initial_rebase": FIRST_REBASE,
        "repair_head": REPAIR_HEAD,
        "prelatest_checkpoint": CHECKPOINT,
        "latest_rebase": LATEST_REBASE,
        "reader_repair_paths": READER_REPAIRS,
        "head": final_head,
        "dev": final_dev,
        "surface": surface,
        "package": output.name,
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "reproduction": "Run this exact helper with the recorded head, dev and surface JSON. Canonical initial diffs and per-path mappings are immutable Task6 artifacts.",
    }
    (SDD / ("task-6-review-package-" + final_head[:12] + ".json")).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
