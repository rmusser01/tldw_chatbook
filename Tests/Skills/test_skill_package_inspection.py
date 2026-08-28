"""Bounded, side-effect-free skill package classification."""

from __future__ import annotations

import io
import os
import warnings
import zipfile

import pytest

from tldw_chatbook.Skills_Interop.skill_package_inspection import (
    FRAMEWORK_MESSAGE,
    FRAMEWORK_RECOVERY_ACTIONS,
    SkillPackageKind,
    inspect_skill_directory,
    inspect_skill_zip,
)
from tldw_chatbook.tldw_api.skills_schemas import (
    MAX_SUPPORTING_FILE_BYTES,
    MAX_SUPPORTING_FILES_TOTAL_BYTES,
)


def _zip(entries: list[tuple[str, str]], *, wrapper: str = "repo-1/") -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, content in entries:
                archive.writestr(wrapper + name, content)
    return buffer.getvalue()


def test_root_skill_has_one_shared_classification_for_directory_and_zip(tmp_path):
    (tmp_path / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")

    directory = inspect_skill_directory(tmp_path)
    archive = inspect_skill_zip(
        _zip([("SKILL.md", "---\nname: demo\n---\n")]),
        repository_source=True,
    )

    assert directory.kind is SkillPackageKind.ROOT_SKILL
    assert archive.kind is SkillPackageKind.ROOT_SKILL
    assert directory.candidates == archive.candidates == ("",)


def test_multiple_candidates_are_stable_deduplicated_and_wrapper_relative(tmp_path):
    for relative in ("skills/zeta", "skills/alpha"):
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text("body", encoding="utf-8")

    directory = inspect_skill_directory(tmp_path)
    archive = inspect_skill_zip(
        _zip(
            [
                ("skills/zeta/SKILL.md", "body"),
                ("skills/alpha/SKILL.md", "body"),
                ("skills/alpha/SKILL.md", "duplicate"),
            ]
        ),
        repository_source=True,
    )

    expected = ("skills/alpha", "skills/zeta")
    assert directory.kind is SkillPackageKind.MULTI_SKILL_REPOSITORY
    assert archive.kind is SkillPackageKind.MULTI_SKILL_REPOSITORY
    assert directory.candidates == archive.candidates == expected


def test_nonempty_repository_without_skill_is_framework_but_direct_zip_is_not(tmp_path):
    (tmp_path / "README.md").write_text("framework", encoding="utf-8")
    archive = _zip([("README.md", "framework")])

    directory = inspect_skill_directory(tmp_path)
    repository = inspect_skill_zip(archive, repository_source=True)
    direct = inspect_skill_zip(archive, repository_source=False)

    for result in (directory, repository):
        assert result.kind is SkillPackageKind.FRAMEWORK_REPOSITORY
        assert result.message == FRAMEWORK_MESSAGE
        assert result.recovery_actions == FRAMEWORK_RECOVERY_ACTIONS
    assert direct.kind is SkillPackageKind.MALFORMED_OR_UNSUPPORTED


@pytest.mark.parametrize("payload", [b"", b"not a zip"])
def test_empty_or_corrupt_archive_is_malformed(payload):
    result = inspect_skill_zip(payload, repository_source=True)
    assert result.kind is SkillPackageKind.MALFORMED_OR_UNSUPPORTED
    assert result.candidates == ()


def test_unsafe_candidate_path_is_malformed():
    result = inspect_skill_zip(
        _zip([("skills/../evil/SKILL.md", "body")]),
        repository_source=True,
    )
    assert result.kind is SkillPackageKind.MALFORMED_OR_UNSUPPORTED


def test_symlinked_directory_skill_is_not_accepted(tmp_path):
    target = tmp_path / "outside.md"
    target.write_text("body", encoding="utf-8")
    skill = tmp_path / "skill"
    skill.mkdir()
    try:
        os.symlink(target, skill / "SKILL.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    result = inspect_skill_directory(tmp_path)
    assert result.kind is SkillPackageKind.MALFORMED_OR_UNSUPPORTED


def test_more_than_display_cap_remains_multi_skill_with_bounded_candidates():
    result = inspect_skill_zip(
        _zip(
            [
                (f"skills/s{index:02d}/SKILL.md", "body")
                for index in range(21)
            ]
        ),
        repository_source=True,
    )

    assert result.kind is SkillPackageKind.MULTI_SKILL_REPOSITORY
    assert result.candidates == tuple(
        f"skills/s{index:02d}" for index in range(20)
    )


@pytest.mark.parametrize(
    "entries",
    [
        [
            ("SKILL.md", "body"),
            ("huge.bin", b"x" * (MAX_SUPPORTING_FILE_BYTES + 1)),
        ],
        [
            ("SKILL.md", "body"),
            ("one.bin", b"x" * (MAX_SUPPORTING_FILES_TOTAL_BYTES // 2 + 1)),
            ("two.bin", b"x" * (MAX_SUPPORTING_FILES_TOTAL_BYTES // 2 + 1)),
        ],
    ],
)
def test_archive_declared_size_caps_reject_misleading_root(entries):
    result = inspect_skill_zip(_zip(entries), repository_source=True)
    assert result.kind is SkillPackageKind.MALFORMED_OR_UNSUPPORTED
