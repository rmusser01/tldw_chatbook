"""Subject snapshot builders: directory + store shapes, digest, manifests."""
from pathlib import Path

import pytest

from tldw_chatbook.Evals.skill_eval.subject import (
    SubjectError, parse_front_matter, subject_from_directory, subject_from_store,
)

SKILL_MD = """---
name: csv-cleaner
description: Use when tidying messy CSV exports before import.
allowed_tools: fs_read fs_list
---

# CSV cleaner

Read the file, then apply `references/rules.md`.
"""


def _make_dir(tmp_path: Path) -> Path:
    d = tmp_path / "csv-cleaner"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (d / "references" / "rules.md").write_text("rules", encoding="utf-8")
    return d


def test_parse_front_matter_splits_metadata_and_body():
    meta, body = parse_front_matter(SKILL_MD)
    assert meta["name"] == "csv-cleaner"
    assert "allowed_tools" in meta
    assert body.startswith("# CSV cleaner")


def test_directory_subject_snapshot(tmp_path):
    subj = subject_from_directory(_make_dir(tmp_path))
    assert subj.name == "csv-cleaner"
    assert subj.source_kind == "directory"
    assert subj.allowed_tools == ("fs_read", "fs_list")
    assert "references/rules.md" in subj.referenced_files
    assert "references/rules.md" in subj.bundle_paths
    assert subj.digest
    assert subj.line_count > 3


def test_directory_subject_missing_skill_md_raises(tmp_path):
    with pytest.raises(SubjectError):
        subject_from_directory(tmp_path)


class _FakeService:
    def __init__(self, resp):
        self._resp = resp

    async def get_skill(self, name):
        if name != self._resp["name"]:
            raise KeyError(name)
        return self._resp


@pytest.mark.asyncio
async def test_store_subject_snapshot():
    resp = {
        "name": "csv-cleaner",
        "description": "Use when tidying messy CSV exports before import.",
        "content": SKILL_MD,
        "bundle_files": [{"path": "references/rules.md", "size": 5,
                          "executable": False, "is_text": True}],
        "trust_status": "trusted",
        "record_id": "r1",
    }
    subj = await subject_from_store(_FakeService(resp), "csv-cleaner")
    assert subj.source_kind == "store"
    assert subj.trust_status == "trusted"
    assert subj.bundle_paths == ("references/rules.md",)
