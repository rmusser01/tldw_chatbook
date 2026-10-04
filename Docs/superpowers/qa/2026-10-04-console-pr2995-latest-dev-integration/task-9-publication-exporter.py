"""Publish exact closed Task6/7/8/9/10/11/12/13 evidence into a fresh QA directory."""

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path


ROOT = Path("/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook")
SDD = ROOT / ".superpowers/sdd/2026-10-03-console-pr2995-review-and-merge"
OUT = ROOT / "Docs/superpowers/qa/2026-10-04-console-pr2995-latest-dev-integration"
SUFFIXES = {".md", ".json", ".jsonl", ".log", ".py", ".xml", ".txt", ".diff"}
KNOWN = set()


def collect(value, key=""):
    if isinstance(value, dict):
        for name, child in value.items():
            collect(child, name)
    elif isinstance(value, list):
        for child in value:
            collect(child, key)
    elif (
        isinstance(value, str)
        and re.search(r"(api.?key|token|password|secret)", key, re.I)
        and not key.endswith(("_env_var", "_env"))
    ):
        if len(value) >= 12 and not re.search(
            r"(<.*>|YOUR_|HERE|example|placeholder)", value, re.I
        ):
            KNOWN.add(value.encode())


def main():
    assert OUT.is_dir(), "Create and review the phase README before exporting."
    assert (OUT / "README.md").is_file()
    selected = sorted(
        p
        for p in SDD.iterdir()
        if p.is_file()
        and not p.is_symlink()
        and p.name.startswith(("task-6", "task-7", "task-8", "task-9", "task-10", "task-11", "task-12", "task-13"))
        and p.suffix in SUFFIXES
    )
    # Only named, already isolated child reports from immutable manifests.
    children = {}
    provider_rows = json.loads(
        (SDD / "task-6-private-child-report-manifest.json").read_text()
    )
    reader_rows = json.loads(
        (SDD / "task-6-reader-evidence-manifest.json").read_text()
    )["files"]
    for row in provider_rows:
        children[row["retained"]] = row["sha256"]
    for row in reader_rows:
        if "/" in row["path"]:
            children[row["path"]] = row["sha256"]
    schema_rows = json.loads(
        (SDD / "task-6-schema-safe-evidence-manifest.json").read_text()
    )["copies"]
    for row in schema_rows:
        children[row["copy"]] = row["sha256"]
    for manifest_name in (
        "task-6-modelconfig-safe-evidence-manifest.json",
        "task-6-screen-cap-safe-evidence-manifest.json",
        "task-7-safe-evidence-manifest.json",
        "task-8-safe-evidence-manifest.json",
        "task-8-fix1-safe-evidence-manifest.json",
    ):
        payload = json.loads((SDD / manifest_name).read_text())
        rows = payload["files"] if isinstance(payload, dict) else payload
        for row in rows:
            relative = str(Path(row["copy"]).relative_to(SDD))
            children[relative] = row["sha256"]
    task10 = json.loads((SDD / "task-10-safe-evidence-map.json").read_text())
    for row in task10["files"].values():
        children[row["path"]] = row["sha256"]
    for manifest_name in (
        "task-9-safe-evidence-manifest.json",
        "task-9-qualification-extension-safe-evidence-manifest.json",
        "task-9-final-qualification-safe-evidence-manifest.json",
        "task-9-remaining-fixture-safe-evidence-manifest.json",
        "task-9-summary-native-qualification-safe-evidence-manifest.json",
        "task-9-video-observer-safe-evidence-manifest.json",
        "task-9-root-final-loading-safe-evidence-manifest.json",
        "task-11-safe-evidence-manifest.json",
        "task-11-newest-dev-safe-evidence-manifest.json",
        "task-12-safe-evidence-manifest.json",
        "task-13-safe-evidence-manifest.json",
    ):
        payload = json.loads((SDD / manifest_name).read_text())
        for relative, row in payload["files"].items():
            children[relative] = row if isinstance(row, str) else row["sha256"]
    for relative, expected in sorted(children.items()):
        parts = Path(relative).parts
        if len(parts) == 1:
            path = SDD / relative
            assert path in selected and path.is_file() and not path.is_symlink()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            continue
        assert len(parts) >= 2 and ".." not in parts and not Path(relative).is_absolute() and parts[0] in {
            "task-6-private-child-reports",
            "task-6-reader-child-reports",
            "task-6-schema-safe-evidence",
            "task-6-modelconfig-safe-evidence",
            "task-6-screen-cap-safe-evidence",
            "task-7-safe-evidence",
            "task-8-safe-evidence",
            "task-8-fix1-safe-evidence",
            "task-9-safe-evidence",
            "task-9-qualification-extension-safe-evidence",
            "task-9-final-qualification-safe-evidence",
            "task-9-remaining-fixture-safe-evidence",
            "task-9-summary-native-qualification-safe-evidence",
            "task-9-video-observer-safe-evidence",
            "task-9-root-final-loading-safe-evidence",
            "task-10-safe-evidence",
            "task-11-safe-evidence",
            "task-11-newest-dev-safe-evidence",
            "task-12-safe-evidence",
            "task-13-safe-evidence",
        }, relative
        path = SDD / relative
        allowed_suffixes = (
            {".json", ".log", ".xml", ".md"}
            if parts[0]
            in {
                "task-6-schema-safe-evidence",
                "task-6-modelconfig-safe-evidence",
                "task-6-screen-cap-safe-evidence",
                "task-7-safe-evidence",
                "task-8-safe-evidence",
                "task-8-fix1-safe-evidence",
                "task-9-safe-evidence",
                "task-9-qualification-extension-safe-evidence",
                "task-9-final-qualification-safe-evidence",
                "task-9-remaining-fixture-safe-evidence",
            "task-9-summary-native-qualification-safe-evidence",
            "task-9-video-observer-safe-evidence",
                "task-9-root-final-loading-safe-evidence",
                "task-10-safe-evidence",
                "task-11-safe-evidence",
                "task-11-newest-dev-safe-evidence",
            "task-12-safe-evidence",
            "task-13-safe-evidence",
            }
            else {".log", ".xml"}
        )
        if parts[0] in {"task-12-safe-evidence", "task-13-safe-evidence"}:
            allowed_suffixes.add(".txt")  # Exact manifest-named source verifier text.
        if parts[0] in {"task-12-safe-evidence", "task-13-safe-evidence"}:
            allowed_suffixes.add(".patch")  # Exact manifest-named formatter/cap overlay.
        assert path.suffix in allowed_suffixes, (relative, sorted(allowed_suffixes))
        assert path.is_file() and path.resolve().is_relative_to(SDD.resolve())
        assert all(not ancestor.is_symlink() for ancestor in [path, *path.parents] if ancestor.is_relative_to(SDD))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        selected.append(path)
    collect(
        tomllib.loads(
            Path("/Users/macbook-dev/.config/tldw_cli/config.toml").read_text()
        )
    )
    for name, value in os.environ.items():
        if (
            re.search(r"(api.?key|token|password|secret)", name, re.I)
            and len(value) >= 12
        ):
            KNOWN.add(value.encode())
    patterns = [
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}"),
        re.compile(rb"(?<![A-Za-z0-9_])sk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}"),
        re.compile(rb"AKIA[0-9A-Z]{16}"),
    ]
    public_literals = set()
    classification_sources = []
    for rel, constant in (
        ("tldw_chatbook/config.py", "TLDW_API_PLACEHOLDER_AUTH_TOKEN"),
        ("Tests/UI/test_settings_encryption_card.py", "PLAINTEXT_KEY"),
        ("Tests/test_config_encryption_lifecycle.py", "PLAINTEXT_KEY"),
    ):
        source = (ROOT / rel).read_bytes()
        candidates = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == constant for target in targets):
                    assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                    candidates.append(node.value.value)
        assert len(candidates) == 1
        value = candidates[0]
        if constant == "PLAINTEXT_KEY":
            assert "plaintext" in value.lower()  # Public, deterministic cipher-test input.
        public_literals.add(value.encode())
        classification_sources.append({"path": rel, "constant": constant, "source_sha256": hashlib.sha256(source).hexdigest(), "category": "public project placeholder" if "PLACEHOLDER" in constant else "public deterministic encryption test fixture"})
    KNOWN.difference_update(public_literals)
    findings = []
    manifest = []
    for path in selected + [OUT / "README.md"]:
        data = path.read_bytes()
        if any(value in data for value in KNOWN):
            findings.append({"path": path.name, "kind": "known-local-credential"})
        if any(match.group() not in public_literals for pattern in patterns for match in pattern.finditer(data)):
            findings.append(
                {"path": path.name, "kind": "recognized-token-or-private-key"}
            )
        manifest.append(
            {
                "path": str(path.relative_to(SDD))
                if path.is_relative_to(SDD)
                else path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    assert not findings, json.dumps(findings)
    for path in selected:
        destination = OUT / path.relative_to(SDD)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert (
            not destination.exists() or destination.read_bytes() == path.read_bytes()
        ), destination.name
        shutil.copyfile(path, destination)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "source_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "files": manifest,
                "selection": "Exact regular Task6/7/8/9/10/11/12/13 top-level evidence and only hash-verified child reports named by immutable manifests; Task12/13 also admit their manifest-named verifier source text. No private profiles/config/databases/cache, symlinks or unrelated plan artifacts. Historical QA retains all original bytes, with one separately mapped exact upstream append.",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "publication-audit.json").write_text(
        json.dumps(
            {
                "files": len(manifest),
                "bytes": sum(row["bytes"] for row in manifest),
                "known_local_credential_values_compared": len(KNOWN),
                "known_matches": 0,
                "public_literal_classification": classification_sources,
                "recognized_token_or_private_key_findings": 0,
                "private_profiles_exported": False,
                "limit": "Exact known-value and recognizable-token checks over selected bytes, with only three provenance-pinned public project/fixture literals classified separately; not a general security audit.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "files": len(manifest),
                "bytes": sum(row["bytes"] for row in manifest),
                "audit_findings": 0,
                "private_profiles_exported": False,
            }
        )
    )


if __name__ == "__main__":
    main()
