"""Static guard for legacy server-client builder migration audit drift."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "Docs/Development/server-client-provider-migration-audit.md"
SOURCE_ROOT = REPO_ROOT / "tldw_chatbook"

DIRECT_BUILDER_RE = re.compile(
    r"\b(?:build_runtime_api_client_from_config|build_runtime_api_client|build_tldw_api_client_from_config)\s*\("
)
INDIRECT_BUILDER_RE = re.compile(
    r"\b(?:"
    r"build_server_chatbook_service|"
    r"build_server_chatbook_service_from_config|"
    r"Server(?:ChatConversation|ChatLoop|CharacterPersona|ChatDictionary|MediaReading|NotesWorkspace|Prompt|Chatbook|PromptStudio)Service\.from_config"
    r")\s*\("
)
AUDIT_ROW_RE = re.compile(
    r"^\|\s*`(?P<path>tldw_chatbook/[^`]+)`\s*\|\s*(?P<lines>[^|]+?)\s*\|\s*(?P<notes>.+?)\s*\|$",
    re.MULTILINE,
)
LINE_TOKEN_RE = re.compile(r"\b(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\b")


def _normalize_builder_line(line: str) -> str:
    return " ".join(line.strip().split())


def _extract_semantic_matches(notes: str) -> Counter[str]:
    marker = re.search(r"Semantic match(?:es)?:\s*", notes)
    if not marker:
        return Counter()

    snippets = Counter()
    for snippet in re.findall(r"`([^`]+)`", notes[marker.end() :]):
        normalized = _normalize_builder_line(snippet)
        if DIRECT_BUILDER_RE.search(normalized) or INDIRECT_BUILDER_RE.search(normalized):
            snippets[normalized] += 1
    return snippets


def _audited_match_metadata(audit_path: Path = AUDIT_PATH) -> dict[str, dict[str, object]]:
    audit_text = audit_path.read_text(encoding="utf-8")
    allowed: dict[str, dict[str, object]] = {}
    for row in AUDIT_ROW_RE.finditer(audit_text):
        path = row.group("path")
        count = 0
        for token in LINE_TOKEN_RE.finditer(row.group("lines")):
            start = int(token.group("start"))
            end = int(token.group("end") or start)
            count += end - start + 1
        metadata = allowed.setdefault(path, {"count": 0, "semantic_matches": Counter()})
        metadata["count"] = int(metadata["count"]) + count
        metadata["semantic_matches"].update(_extract_semantic_matches(row.group("notes")))
    return allowed


def _builder_matches(source_root: Path = SOURCE_ROOT, repo_root: Path = REPO_ROOT) -> set[tuple[str, int, str]]:
    matches: set[tuple[str, int, str]] = set()
    for source_path in sorted(source_root.rglob("*.py")):
        relative_path = source_path.relative_to(repo_root).as_posix()
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            if DIRECT_BUILDER_RE.search(line) or INDIRECT_BUILDER_RE.search(line):
                matches.add((relative_path, line_number, _normalize_builder_line(line)))
    return matches


def _audit_drift(audit_path: Path = AUDIT_PATH, source_root: Path = SOURCE_ROOT, repo_root: Path = REPO_ROOT) -> list[str]:
    audited_metadata = _audited_match_metadata(audit_path=audit_path)
    matches_by_path: dict[str, list[tuple[int, str]]] = {}
    for path, line_number, line in sorted(_builder_matches(source_root=source_root, repo_root=repo_root)):
        matches_by_path.setdefault(path, []).append((line_number, line))

    drift: list[str] = []
    for path in sorted(set(audited_metadata) | set(matches_by_path)):
        matches = matches_by_path.get(path, [])
        metadata = audited_metadata.get(path, {"count": 0, "semantic_matches": Counter()})
        audited_count = int(metadata["count"])
        actual_count = len(matches)
        if actual_count <= audited_count:
            actual_semantics = Counter(line for _, line in matches)
            audited_semantics = metadata["semantic_matches"]
            if actual_semantics == audited_semantics:
                continue

            missing = list((actual_semantics - audited_semantics).elements())
            extra = list((audited_semantics - actual_semantics).elements())
            drift.append(
                f"{path}: semantic match drift; missing={missing or '[]'} extra={extra or '[]'}"
            )
            continue

        drift.append(
            f"{path}: audited {audited_count} builder call(s), found {actual_count}: "
            + "; ".join(f"{line_number}: {line}" for line_number, line in matches)
        )

    return drift


def test_audit_guard_rejects_new_unlisted_legacy_builder(tmp_path: Path):
    repo_root = tmp_path / "repo"
    source_root = repo_root / "tldw_chatbook"
    source_root.mkdir(parents=True)
    audit_path = repo_root / "Docs/Development/server-client-provider-migration-audit.md"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        "| Module | Audit lines | Notes |\n| --- | ---: | --- |\n",
        encoding="utf-8",
    )
    (source_root / "example.py").write_text(
        "build_runtime_api_client_from_config(app_config)\n",
        encoding="utf-8",
    )

    drift = _audit_drift(audit_path=audit_path, source_root=source_root, repo_root=repo_root)

    assert drift
    assert "example.py" in drift[0]


def test_audit_guard_uses_semantic_not_line_number_matching(tmp_path: Path):
    repo_root = tmp_path / "repo"
    source_root = repo_root / "tldw_chatbook"
    source_root.mkdir(parents=True)
    audit_path = repo_root / "Docs/Development/server-client-provider-migration-audit.md"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        "\n".join(
            [
                "| Module | Audit lines | Notes |",
                "| --- | ---: | --- |",
                "| `tldw_chatbook/example.py` | 1 | Semantic match: `build_runtime_api_client(app_config)`. |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_root / "example.py").write_text(
        "\n\nbuild_runtime_api_client_from_config(app_config)\n",
        encoding="utf-8",
    )

    drift = _audit_drift(audit_path=audit_path, source_root=source_root, repo_root=repo_root)

    assert drift == [
        "tldw_chatbook/example.py: semantic match drift; missing=['build_runtime_api_client_from_config(app_config)'] extra=['build_runtime_api_client(app_config)']"
    ]


def test_audit_guard_rejects_stale_audited_row_with_no_live_match(tmp_path: Path):
    repo_root = tmp_path / "repo"
    source_root = repo_root / "tldw_chatbook"
    source_root.mkdir(parents=True)
    audit_path = repo_root / "Docs/Development/server-client-provider-migration-audit.md"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        "\n".join(
            [
                "| Module | Audit lines | Notes |",
                "| --- | ---: | --- |",
                "| `tldw_chatbook/example.py` | 1 | Semantic match: `build_runtime_api_client_from_config(app_config)`. |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_root / "example.py").write_text(
        "def no_builder_here():\n    return None\n",
        encoding="utf-8",
    )

    drift = _audit_drift(audit_path=audit_path, source_root=source_root, repo_root=repo_root)

    assert drift == [
        "tldw_chatbook/example.py: semantic match drift; missing=[] extra=['build_runtime_api_client_from_config(app_config)']"
    ]


def test_legacy_server_client_builder_matches_are_listed_in_migration_audit():
    drift = _audit_drift()

    assert drift == []
