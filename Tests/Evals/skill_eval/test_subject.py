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


# Qodo F3: the skills store's canonical front-matter spelling is the
# hyphenated ``allowed-tools``; a directory skill using it used to
# snapshot as tool-less.
def test_directory_subject_accepts_hyphenated_allowed_tools(tmp_path):
    skill_md = SKILL_MD.replace("allowed_tools: fs_read fs_list",
                                "allowed-tools: fs_read fs_list")
    d = tmp_path / "csv-cleaner"
    d.mkdir()
    (d / "SKILL.md").write_text(skill_md, encoding="utf-8")
    subj = subject_from_directory(d)
    assert subj.allowed_tools == ("fs_read", "fs_list")


def test_underscore_spelling_wins_over_hyphenated(tmp_path):
    both = SKILL_MD.replace(
        "allowed_tools: fs_read fs_list\n",
        "allowed_tools: fs_read\nallowed-tools: fs_write\n")
    d = tmp_path / "csv-cleaner"
    d.mkdir()
    (d / "SKILL.md").write_text(both, encoding="utf-8")
    subj = subject_from_directory(d)
    assert subj.allowed_tools == ("fs_read",)


@pytest.mark.asyncio
async def test_store_subject_prefers_the_normalized_record_field():
    # Qodo F3: the get_skill response carries the service-normalized
    # ``allowed_tools`` (built from EITHER front-matter spelling) -- when
    # present it wins over re-parsing the raw front matter.
    resp = {
        "name": "csv-cleaner",
        "description": "Use when tidying messy CSV exports before import.",
        "content": SKILL_MD.replace("allowed_tools: fs_read fs_list",
                                    "allowed-tools: fs_read fs_list"),
        "allowed_tools": ["fs_read", "fs_list", "fs_grep"],
        "bundle_files": [],
        "trust_status": "trusted",
        "record_id": "r1",
    }
    subj = await subject_from_store(_FakeService(resp), "csv-cleaner")
    assert subj.allowed_tools == ("fs_read", "fs_list", "fs_grep")


@pytest.mark.asyncio
async def test_store_subject_falls_back_to_hyphenated_front_matter():
    # Thinner service responses (no normalized field) still resolve the
    # canonical hyphenated spelling from the raw content.
    resp = {
        "name": "csv-cleaner",
        "description": "d",
        "content": SKILL_MD.replace("allowed_tools: fs_read fs_list",
                                    "allowed-tools: fs_read fs_list"),
        "bundle_files": [],
        "trust_status": "unknown",
    }
    subj = await subject_from_store(_FakeService(resp), "csv-cleaner")
    assert subj.allowed_tools == ("fs_read", "fs_list")


# Qodo F15: the user-supplied directory goes through path_validation --
# relative and traversal spellings are rejected as SubjectError, and only
# the validator-returned (resolved) path is read.
def test_directory_subject_rejects_relative_path():
    with pytest.raises(SubjectError):
        subject_from_directory("csv-cleaner")


def test_directory_subject_rejects_traversal_path():
    with pytest.raises(SubjectError):
        subject_from_directory("../../etc")


def test_directory_subject_rejects_missing_directory(tmp_path):
    with pytest.raises(SubjectError):
        subject_from_directory(tmp_path / "no-such-dir")


def test_directory_subject_reads_the_resolved_validator_path(tmp_path):
    # A symlinked spelling still snapshots (the validator resolves it) --
    # what is rejected is a path that does not name a real directory.
    real = _make_dir(tmp_path)
    link = tmp_path / "link-to-skill"
    link.symlink_to(real, target_is_directory=True)
    subj = subject_from_directory(link)
    assert subj.name == "csv-cleaner"
    assert subj.source_kind == "directory"
