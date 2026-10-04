"""Check that a proposed ADR220 filename has no existing canonical claim."""

import json
import re
import subprocess
from pathlib import Path


ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
EXPECTED = "220-console-human-decision-coordination-ownership.md"


def call(*argv):
    return subprocess.check_output(argv, cwd=ROOT, text=True)


def main():
    output = SDD / "task-9-adr220-allocation-recheck.json"
    assert not output.exists()
    refs = call("git", "for-each-ref", "--format=%(refname) %(objectname)").splitlines()
    objects = sorted({line.rsplit(" ", 1)[1] for line in refs})
    ref_claims = []
    for oid in objects:
        names = call(
            "git", "ls-tree", "-r", "--name-only", oid, "--", "backlog/decisions"
        ).splitlines()
        for name in names:
            if re.match(r"220-", Path(name).name):
                ref_claims.append({"object": oid, "path": name})
    worktree_claims = []
    for block in call("git", "worktree", "list", "--porcelain").split("\n\n"):
        line = next(
            (line for line in block.splitlines() if line.startswith("worktree ")), None
        )
        if line:
            folder = Path(line.removeprefix("worktree ")) / "backlog/decisions"
            worktree_claims.extend(
                {"worktree": str(folder.parent.parent), "name": p.name}
                for p in folder.glob("220-*.md")
            )
    prs = json.loads(
        call(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "repos/rmusser01/tldw_chatbook/pulls?state=open&per_page=100",
        )
    )
    pr_claims = []
    examined = []
    for pr in (pr for page in prs for pr in page):
        pages = json.loads(
            call(
                "gh",
                "api",
                "--paginate",
                "--slurp",
                "repos/rmusser01/tldw_chatbook/pulls/"
                + str(pr["number"])
                + "/files?per_page=100",
            )
        )
        names = [item["filename"] for page in pages for item in page]
        examined.append(
            {"number": pr["number"], "head": pr["head"]["sha"], "files": len(names)}
        )
        pr_claims.extend(
            {"number": pr["number"], "path": name}
            for name in names
            if name.startswith("backlog/decisions/220-")
        )
    foreign = [
        row
        for row in ref_claims + worktree_claims + pr_claims
        if Path(row.get("path", row.get("name", ""))).name != EXPECTED
    ]
    result = {
        "refs": len(refs),
        "objects": len(objects),
        "ref_claims": ref_claims,
        "worktree_filename_claims": worktree_claims,
        "open_prs": examined,
        "open_pr_claims": pr_claims,
        "foreign_220_claims": foreign,
        "expected_own_claim": EXPECTED,
        "current_dev": json.loads(
            call("gh", "api", "repos/rmusser01/tldw_chatbook/git/ref/heads/dev")
        )["object"]["sha"],
        "limit": "Canonical filename identity scan; working-tree inventory reads filenames only and never unrelated SDD or private source bodies.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    assert not (ref_claims or worktree_claims or pr_claims), (
        "Foreign220 claim: inspect the recorded paths before publication."
    )
    print(
        json.dumps(
            {
                "refs": len(refs),
                "objects": len(objects),
                "open_prs": len(examined),
                "foreign_claims": len(foreign),
                "dev": result["current_dev"],
            }
        )
    )


if __name__ == "__main__":
    main()
