"""Static guard for legacy server-client builder migration audit drift."""

from __future__ import annotations

import re
from pathlib import Path


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
AUDIT_ROW_RE = re.compile(r"^\|\s*`(?P<path>tldw_chatbook/[^`]+)`\s*\|\s*(?P<lines>[^|]+?)\s*\|", re.MULTILINE)
LINE_TOKEN_RE = re.compile(r"\b(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\b")


def _audited_match_counts() -> dict[str, int]:
    audit_text = AUDIT_PATH.read_text(encoding="utf-8")
    allowed: dict[str, int] = {}
    for row in AUDIT_ROW_RE.finditer(audit_text):
        path = row.group("path")
        count = 0
        for token in LINE_TOKEN_RE.finditer(row.group("lines")):
            start = int(token.group("start"))
            end = int(token.group("end") or start)
            count += end - start + 1
        allowed[path] = allowed.get(path, 0) + count
    return allowed


def _builder_matches() -> set[tuple[str, int, str]]:
    matches: set[tuple[str, int, str]] = set()
    for source_path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative_path = source_path.relative_to(REPO_ROOT).as_posix()
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            if DIRECT_BUILDER_RE.search(line) or INDIRECT_BUILDER_RE.search(line):
                matches.add((relative_path, line_number, line.strip()))
    return matches


def test_legacy_server_client_builder_matches_are_listed_in_migration_audit():
    audited_counts = _audited_match_counts()
    matches_by_path: dict[str, list[tuple[int, str]]] = {}
    for path, line_number, line in sorted(_builder_matches()):
        matches_by_path.setdefault(path, []).append((line_number, line))

    drift: list[str] = []
    for path, matches in sorted(matches_by_path.items()):
        audited_count = audited_counts.get(path, 0)
        actual_count = len(matches)
        if actual_count <= audited_count:
            continue
        drift.append(
            f"{path}: audited {audited_count} builder call(s), found {actual_count}: "
            + "; ".join(f"{line_number}: {line}" for line_number, line in matches)
        )

    assert drift == []
