"""Verify the source handoff and safe evidence without replaying tests."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
HEAD = "40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77"


def git(*args):
    return subprocess.check_output(["git", "-c", "gc.auto=0", *args], cwd=ROOT)


def tree(revision):
    rows = {}
    for item in git("ls-tree", "-r", "-z", "--full-tree", revision).split(b"\0"):
        if item:
            meta, name = item.split(b"\t", 1)
            mode, kind, blob = meta.decode().split()
            rows[name.decode()] = blob
    return rows


def blobs(ids):
    ids = sorted(set(ids))
    data = (
        git("cat-file", "--batch")
        if not ids
        else subprocess.check_output(
            ["git", "-c", "gc.auto=0", "cat-file", "--batch"],
            cwd=ROOT,
            input=("\n".join(ids) + "\n").encode(),
        )
    )
    results = {}
    offset = 0
    for expected in ids:
        end = data.index(b"\n", offset)
        oid, kind, size = data[offset:end].decode().split()
        assert oid == expected and kind == "blob"
        size = int(size)
        start = end + 1
        results[oid] = data[start : start + size]
        assert data[start + size : start + size + 1] == b"\n"
        offset = start + size + 1
    assert offset == len(data)
    return results


def main():
    preservation = json.loads((SDD / "task-8-final-preservation.json").read_text())
    freeze = json.loads((SDD / "task-8-final-freeze-map.json").read_text())
    safe = json.loads((SDD / "task-8-safe-evidence-manifest.json").read_text())
    final_tree = tree(HEAD)
    quoted_tree = {}
    for item in (
        git("-c", "core.quotepath=true", "ls-tree", "-r", "--full-tree", HEAD)
        .decode()
        .splitlines()
    ):
        meta, name = item.split("\t", 1)
        quoted_tree[name] = meta.split()[2]
    assert quoted_tree == freeze["final_tree"]
    assert len(quoted_tree) == len(final_tree)
    assert not git("status", "--porcelain")
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    for name in ("historical_qa", "incoming_qa"):
        rows = preservation[name]
        assert not rows["mismatches"] and len(rows["blobs"]) == rows["count"]
        assert all(final_tree[p] == b for p, b in rows["blobs"].items())
    required = freeze["required_historical_qa"]
    assert len(required) == 11572
    assert all(
        r["required_blob"] == r["final_blob"] == final_tree[r["path"]] for r in required
    )
    paths = {r["path"] for r in preservation["source_and_test_maps"]}
    paths.update(r["path"] for r in freeze["task7_changed_function_carry"])
    data = blobs(final_tree[p] for p in paths)
    assert all(
        hashlib.sha256(data[final_tree[r["path"]]]).hexdigest() == r["final"]["sha256"]
        for r in preservation["source_and_test_maps"]
    )
    for row in freeze["task7_changed_function_carry"]:
        module = ast.parse(data[final_tree[row["path"]]])
        stem = row["task7_name"].rsplit("#", 1)[0].rsplit(".", 1)[-1]
        matches = [
            n
            for n in ast.walk(module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == stem
        ]
        assert row["present_exact_in_final"]
        assert any(
            hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            == row["ast_sha256"]
            for n in matches
        )
    for row in safe["files"]:
        path = Path(row["copy"])
        assert path.is_relative_to(SDD / "task-8-safe-evidence")
        assert (
            path.is_file()
            and not path.is_symlink()
            and path.suffix in {".json", ".log", ".xml"}
        )
        content = path.read_bytes()
        assert (
            len(content) == row["bytes"]
            and hashlib.sha256(content).hexdigest() == row["sha256"]
        )
    qualification = json.loads(
        (SDD / "task-8-final-startup-navigation.json").read_text()
    )
    assert qualification["exit"] == 0 and not qualification["terminated"]
    assert qualification["source_before"] == qualification["source_after"]
    formats = json.loads((SDD / "task-8-final-format-proof.json").read_text())
    formats += json.loads((SDD / "task-8-format-proof.json").read_text())
    verified = []
    for p, sha in qualification["source_after"].items():
        final = hashlib.sha256(git("show", f"{HEAD}:{p}")).hexdigest()
        steps = []
        while sha != final:
            links = [
                r
                for r in formats
                if r["path"] == p
                and r["before_sha256"] == sha
                and r["full_module_ast_exact"]
            ]
            assert len(links) == 1, (p, "missing unique format carry")
            sha = links[0]["after_sha256"]
            steps.append({"before": links[0]["before_sha256"], "after": sha})
            assert len(steps) <= len(formats)
        verified.append({"path": p, "final_sha256": final, "format_carry": steps})
    result = {
        "head": HEAD,
        "clean_before_root_metadata": True,
        "final_tree_exact": len(final_tree),
        "contemporaneous_selected_final_hashes_verified": len(
            preservation["source_and_test_maps"]
        ),
        "task7_changed_functions_verified": len(freeze["task7_changed_function_carry"]),
        "required_historical_qa_exact": len(required),
        "all_checkpoint_qa_exact": preservation["historical_qa"]["count"],
        "incoming_qa_exact": preservation["incoming_qa"]["count"],
        "safe_explicit_copies_verified": len(safe["files"]),
        "safe_bytes": sum(r["bytes"] for r in safe["files"]),
        "final_qualification_source_before_after_exact": True,
        "final_qualification_format_chains": verified,
        "tests_replayed": False,
        "limits": "Format carry uses explicit contemporaneous before/after hashes and recorded full-module AST equality; raw qualified preformat preimages are not claimed present in Git.",
    }
    output = SDD / "task-8-root-handoff-verification.json"
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"final_qualification_format_chains", "limits"}
            }
        )
    )


if __name__ == "__main__":
    main()
