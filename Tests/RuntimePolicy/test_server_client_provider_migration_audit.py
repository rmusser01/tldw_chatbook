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
    r"Server[A-Za-z]+Service\.from_config"
    r")\s*\("
)
AUDIT_ROW_RE = re.compile(
    r"^\|\s*`(?P<path>tldw_chatbook/[^`]+)`\s*\|\s*(?P<lines>[^|]+?)\s*\|\s*(?P<notes>.+?)\s*\|$",
    re.MULTILINE,
)
LINE_TOKEN_RE = re.compile(r"\b(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\b")
HEADING_RE = re.compile(r"^(?P<marks>#{2,4})\s+(?P<title>.+?)\s*$")


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


def _normalize_reason_category(title: str) -> str:
    return " ".join(title.strip().split()).lower()


def _semantic_entry_key(snippet: str) -> str:
    if snippet.startswith("def "):
        return "signature"
    return "call_pattern"


def _audit_rows(audit_path: Path = AUDIT_PATH) -> list[tuple[str, str, str, str]]:
    audit_text = audit_path.read_text(encoding="utf-8")
    rows: list[tuple[str, str, str, str]] = []
    reason_category = ""

    for line in audit_text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            reason_category = _normalize_reason_category(heading.group("title"))
            continue

        row = AUDIT_ROW_RE.match(line)
        if not row:
            continue

        rows.append((row.group("path"), row.group("lines"), row.group("notes"), reason_category))

    return rows


def load_provider_migration_audit_entries(audit_path: Path = AUDIT_PATH) -> list[dict[str, object]]:
    raw_entries: list[dict[str, object]] = []
    counts_by_path: Counter[str] = Counter()
    for path, _lines, notes, category in _audit_rows(audit_path=audit_path):
        semantic_matches = _extract_semantic_matches(notes)
        for snippet, snippet_count in semantic_matches.items():
            counts_by_path[path] += snippet_count
            for _ in range(snippet_count):
                raw_entries.append(
                    {
                        "path": path,
                        "reason_category": category,
                        _semantic_entry_key(snippet): snippet,
                    }
                )

    return [
        {
            **entry,
            "per_file_match_count": counts_by_path[str(entry["path"])],
        }
        for entry in raw_entries
    ]


def _audited_match_metadata(audit_path: Path = AUDIT_PATH) -> dict[str, dict[str, object]]:
    allowed: dict[str, dict[str, object]] = {}

    for entry in load_provider_migration_audit_entries(audit_path=audit_path):
        path = str(entry["path"])
        semantic_match = entry.get("signature") or entry.get("call_pattern")
        metadata = allowed.setdefault(
            path,
            {
                "count": 0,
                "semantic_matches": Counter(),
                "line_only_rows": 0,
                "reason_categories": Counter(),
            },
        )
        metadata["count"] = int(metadata["count"]) + 1
        metadata["semantic_matches"].update([semantic_match])
        metadata["reason_categories"].update([entry["reason_category"]])

    for path, lines, notes, _category in _audit_rows(audit_path=audit_path):
        if LINE_TOKEN_RE.search(lines) and not _extract_semantic_matches(notes):
            metadata = allowed.setdefault(
                path,
                {
                    "count": 0,
                    "semantic_matches": Counter(),
                    "line_only_rows": 0,
                    "reason_categories": Counter(),
                },
            )
            metadata["line_only_rows"] = int(metadata["line_only_rows"]) + 1
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
        metadata = audited_metadata.get(
            path,
            {"count": 0, "semantic_matches": Counter(), "line_only_rows": 0},
        )
        audited_count = int(metadata["count"])
        actual_count = len(matches)
        if int(metadata["line_only_rows"]):
            drift.append(
                f"{path}: line-only audit row is invalid; add Semantic match snippets"
            )
            continue

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


def test_audit_guard_rejects_new_unlisted_server_service_from_config(tmp_path: Path):
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
        "ServerRAGAdminService.from_config(app_config)\n",
        encoding="utf-8",
    )

    drift = _audit_drift(audit_path=audit_path, source_root=source_root, repo_root=repo_root)

    assert drift
    assert "ServerRAGAdminService.from_config(app_config)" in drift[0]


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


def test_audit_guard_rejects_line_number_only_rows(tmp_path: Path):
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
                "| `tldw_chatbook/example.py` | 1 | Line-only audit row. |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_root / "example.py").write_text(
        "build_runtime_api_client_from_config(app_config)\n",
        encoding="utf-8",
    )

    drift = _audit_drift(audit_path=audit_path, source_root=source_root, repo_root=repo_root)

    assert drift == [
        "tldw_chatbook/example.py: line-only audit row is invalid; add Semantic match snippets"
    ]


def test_audit_guard_rejects_stale_line_number_only_rows(tmp_path: Path):
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
                "| `tldw_chatbook/example.py` | 123 | Line-only stale row. |",
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
        "tldw_chatbook/example.py: line-only audit row is invalid; add Semantic match snippets"
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


def test_audit_guard_accepts_line_number_drift_when_semantic_key_matches(tmp_path: Path):
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
                "| `tldw_chatbook/example.py` | 999 | Informational line hint only. Semantic match: `build_runtime_api_client_from_config(app_config)`. |",
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

    assert drift == []


def test_raw_client_builder_audit_uses_semantic_keys_not_line_numbers():
    entries = load_provider_migration_audit_entries()

    assert entries
    for entry in entries:
        assert entry["path"].startswith("tldw_chatbook/")
        assert entry["reason_category"]
        assert entry["per_file_match_count"] >= 1
        assert "signature" in entry or "call_pattern" in entry
        assert "line" not in entry


def test_raw_client_builder_audit_tracks_reason_category_and_per_file_count(tmp_path: Path):
    repo_root = tmp_path / "repo"
    source_root = repo_root / "tldw_chatbook"
    source_root.mkdir(parents=True)
    audit_path = repo_root / "Docs/Development/server-client-provider-migration-audit.md"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        "\n".join(
            [
                "# Audit",
                "",
                "### Provider-Backed Compatibility Adapter Uses",
                "",
                "| Module | Audit lines | Notes |",
                "| --- | ---: | --- |",
                "| `tldw_chatbook/example.py` | 200 | Informational line hint. Semantic match: `build_runtime_api_client(app_config)`. |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_provider_migration_audit_entries(audit_path=audit_path)

    assert entries == [
        {
            "path": "tldw_chatbook/example.py",
            "reason_category": "provider-backed compatibility adapter uses",
            "per_file_match_count": 1,
            "call_pattern": "build_runtime_api_client(app_config)",
        }
    ]


def test_legacy_server_client_builder_matches_are_listed_in_migration_audit():
    drift = _audit_drift()

    assert drift == []
