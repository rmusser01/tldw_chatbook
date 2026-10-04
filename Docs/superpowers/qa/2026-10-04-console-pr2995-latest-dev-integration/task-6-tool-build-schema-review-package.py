"""Freeze new schema/startup integration and its exact corrective source surface."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
OLD_DEV = "f1f80847a410525ce1b00d27ea0e8be824a98422"
PRE_REBASE = "392655567f7754a394ea1e7aba8d75e81d2cadb2"
REBASE_HEAD = "07740acdbc4954c594fc6130bb20d0e3730a7207"


def git(*args):
    return subprocess.check_output(
        ["git", "-c", "gc.auto=0", *args], cwd=ROOT, text=True
    )


def main():
    final_head, final_dev, incoming_surface_file, final_surface_file = sys.argv[1:]
    incoming = json.loads(Path(incoming_surface_file).read_text())
    final = json.loads(Path(final_surface_file).read_text())
    for surface in (incoming, final):
        assert isinstance(surface, list) and all(
            isinstance(path, str) for path in surface
        )
    for sha in (final_head, final_dev, PRE_REBASE, REBASE_HEAD):
        assert git("rev-parse", sha).strip() == sha
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", final_dev, final_head],
        cwd=ROOT,
        check=True,
    )
    output = SDD / ("task-6-schema-startup-review-" + final_head[:12] + ".diff")
    assert not output.exists()
    pieces = [
        "# Immutable Task6 new schema/startup integration review\n",
        "Earlier source and I1 fix review:27ee390d57504012c6fb49d5f5256a30abb1609d\n",
        "Pre-rebase checkpoint: " + PRE_REBASE + "\n",
        "Combined rebase: " + REBASE_HEAD + "\n",
        "Old dev: " + OLD_DEV + "\n",
        "Final dev: " + final_dev + "\n",
        "Final reviewed source: " + final_head + "\n",
        "\nEarlier feature/Task1-6 and scanner reviews remain bound to their source mappings. "
        "This package covers the new incoming seams, final feature migration/restore contracts, "
        "and all post-rebase corrective edits. Retained unrelated upstream/QA files are "
        "accounted for by exact manifests. Canonical schema captures are verification artifacts.\n",
        "\n## New upstream seams\n",
        git("log", "--format=%H %s", OLD_DEV + ".." + final_dev),
        git("diff", "--stat", OLD_DEV, final_dev, "--", *incoming),
        git("diff", "-U25", OLD_DEV, final_dev, "--", *incoming),
        "\n## Final feature migration/restore integration over current dev\n",
        git("diff", "--stat", final_dev, final_head, "--", *final),
        git("diff", "--find-renames", "-U30", final_dev, final_head, "--", *final),
        "\n## All corrective edits after combined rebase\n",
        git("log", "--format=%H %s", REBASE_HEAD + ".." + final_head),
        git("diff", "--stat", REBASE_HEAD, final_head),
        git("diff", "--find-renames", "-U30", REBASE_HEAD, final_head),
    ]
    output.write_text("".join(pieces))
    receipt = {
        "pre_rebase": PRE_REBASE,
        "rebase": REBASE_HEAD,
        "old_dev": OLD_DEV,
        "dev": final_dev,
        "source": final_head,
        "incoming_surface": incoming,
        "final_surface": final,
        "package": output.name,
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (SDD / ("task-6-schema-review-package-" + final_head[:12] + ".json")).write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
